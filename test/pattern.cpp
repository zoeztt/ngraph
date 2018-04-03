/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
std::shared_ptr<Node> create_reduction(const std::shared_ptr<Node>& node,
                                       const std::string& init_val,
                                       const AxisSet& reduction_axes)
{
    const auto& et = node->get_element_type();
    auto f_A = std::make_shared<op::Parameter>(et, Shape{});
    auto f_B = std::make_shared<op::Parameter>(et, Shape{});
    auto f =
        std::make_shared<Function>(std::make_shared<T>(f_A, f_B), op::ParameterVector{f_A, f_B});

    auto init = std::make_shared<op::Constant>(et, Shape{}, std::vector<std::string>({init_val}));
    return std::make_shared<op::Reduce>(node, init, f, reduction_axes);
}

std::shared_ptr<Node> xla_sum(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
{
    return create_reduction<op::Add>(node, "0", reduction_axes);
}

static std::shared_ptr<Node> construct_constant_node(int n)
{
    return op::Constant::create(element::i32, Shape{}, {n});
}

bool is_zero(std::shared_ptr<Node> reduce_constant)
{
    return is_equal_to_const_value("0", reduce_constant);
}

bool sum_predicate(std::shared_ptr<Node> gn)
{
    NGRAPH_DEBUG << "pred_v2 : looking at " << gn->get_name();
    if (auto r = std::dynamic_pointer_cast<op::Reduce>(gn))
    {
        auto reducee = gn->get_input_op(0);
        auto reduce_constant = gn->get_input_op(1);

        if (!is_zero(reduce_constant))
        {
            return false;
        }

        auto result = r->get_functions()[0]->get_result()->get_input_op(0);
        NGRAPH_DEBUG << "looking at function's result  " << result->get_name();
        if (auto sum = std::dynamic_pointer_cast<op::Add>(result))
        {
            auto parm1 = std::dynamic_pointer_cast<op::Parameter>(sum->get_input_op(0));
            auto parm2 = std::dynamic_pointer_cast<op::Parameter>(sum->get_input_op(1));

            const auto parm_or_nil = [](std::shared_ptr<Node> p) {
                return p ? p->get_name() : std::string("(nil)");
            };
            NGRAPH_DEBUG << "parm1 = " << parm_or_nil(parm1) << " , parm2 = " << parm_or_nil(parm2)
                         << std::endl;
            if (parm1 && parm2 && parm1 != parm2)
            {
                return true;
            }
        }
    }

    return false;
}

std::shared_ptr<pattern::op::Label> construct_sum_pattern() //for the sake of explicitness
{
    return std::make_shared<pattern::op::Label>(element::i32, Shape{}, sum_predicate);
}

static std::shared_ptr<pattern::op::Label> construct_variance_graph()
{
    // construct varaiance
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::Divide>(xmu, N);
    auto variance_label =
        std::make_shared<pattern::op::Label>(variance, nullptr, NodeVector{variance});

    return variance_label;
}

static std::shared_ptr<pattern::op::Label> construct_mean_graph()
{
    //construct mean;
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);
    auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, NodeVector{mean});
    return mean_label;
}

class TestGraphRewrite : public ngraph::pass::GraphRewrite
{
public:
    void construct_multiply_by_one()
    {
        //pattern #1 : a * 1 = a
        auto iconst1 = construct_constant_node(1);
        auto pattern = std::make_shared<pattern::op::Label>(iconst1);

        ngraph::pattern::gr_callback_fn callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_multiply_by_one against "
                         << m.match_root()->get_name();
            assert(m.match_root()->get_input_ops().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_input_ops().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_input_ops().at(const_node_index));
            auto second_node = m.match_root()->get_input_ops().at(const_node_index);
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return false;
            }

            auto const_values = const_node->get_vector<int32_t>();
            bool all_ones =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 1; });

            if (!all_ones)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 1";
                return false;
            }

            ngraph::replace_node(m.match_root(), pattern_map[pattern]);
            return true;
        };

        auto m = make_shared<TestMatcher>(pattern * iconst1, callback);
        this->add_matcher(m);
    }

    void construct_add_zero()
    {
        //pattern #2 : a + 0 = a
        auto iconst0 = construct_constant_node(0);
        auto pattern = std::make_shared<pattern::op::Label>(iconst0);

        auto callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_add_zero against "
                         << m.match_root()->get_name();
            assert(m.match_root()->get_input_ops().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_input_ops().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_input_ops().at(const_node_index));
            auto second_node = m.match_root()->get_input_ops().at(const_node_index);
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return false;
            }

            auto const_values = const_node->get_vector<int>();
            bool all_zeros =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 0; });

            if (!all_zeros)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 0";
                return false;
            }

            ngraph::replace_node(m.match_root(), pattern_map[pattern]);
            return true;
        };

        auto m = make_shared<TestMatcher>(pattern + iconst0, callback);
        this->add_matcher(m);
    }

    void construct_sum()
    {
        auto sum_pattern = construct_sum_pattern();

        ngraph::pattern::gr_callback_fn callback = [](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_sum_pattern against "
                         << m.match_root()->get_name();
            auto reduce = std::dynamic_pointer_cast<op::Reduce>(m.match_root());
            auto reducee = reduce->get_inputs().at(0).get_output().get_node();
            NGRAPH_DEBUG << "reducee = " << reducee->get_name();
            auto sum =
                std::shared_ptr<ngraph::Node>(new op::Sum(reducee, reduce->get_reduction_axes()));

            ngraph::replace_node(m.match_root(), sum);
            return true;
        };

        auto m = make_shared<TestMatcher>(sum_pattern, callback);
        this->add_matcher(m);
    }

    TestGraphRewrite()
        : GraphRewrite()
    {
        construct_multiply_by_one();
        construct_add_zero();
        construct_sum();
    }
};

static void run_passes(pass::Manager& pass_manager,
                       shared_ptr<Node> graph,
                       std::vector<shared_ptr<op::Parameter>> parms)
{
    auto func = make_shared<Function>(graph, op::ParameterVector{parms});
    pass_manager.run_passes(func);
}

TEST(pattern, graph_rewrite)
{
    Shape shape{};
    pass::Manager pass_manager;
    pass_manager.register_pass<TestGraphRewrite>();

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto c = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto graph_a = a + iconst0;
        auto graph_b = b + iconst0;

        auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, graph_a, c, graph_b},
                                            op::ParameterVector{a, b, c});
        pass_manager.run_passes(f);

        ASSERT_TRUE(graph_a->get_output_inputs(0).empty());
        ASSERT_TRUE(graph_b->get_output_inputs(0).empty());

        auto expected = ngraph::NodeVector{a, b, a, c, b};
        ASSERT_TRUE(count_ops_of_type<op::Add>(f) == 0);
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto sum = (a + iconst0);
        auto graph = b + sum;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(sum->get_output_inputs(0)
                        .empty()); //graph's input is removed from sum's output.get_inputs()
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto mul = (a * iconst1);
        auto graph = b + mul;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(mul->get_outputs()
                        .at(0)
                        .get_inputs()
                        .empty()); //graph's input is removed from sum's output.get_inputs()
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto graph = ((((a * iconst1) * iconst1) * iconst1) * iconst1) + b;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(0), a);
        ASSERT_EQ(&graph->get_inputs().at(0).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(0))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto iconst1 = construct_constant_node(1);
        auto graph = b + (iconst0 + ((a + iconst0) * iconst1));
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto graph = b + (iconst1 * (iconst1 * (iconst1 * (iconst1 * a))));
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    //Sum rewrite
    {
        auto parm = make_shared<op::Parameter>(element::i32, Shape{2, 2});
        auto axes = AxisSet{0, 1};
        auto sum_graph = xla_sum(parm, axes);
        auto innermost_abs = make_shared<op::Abs>(sum_graph);

        auto nested_sum_graph = make_shared<op::Abs>(
            make_shared<op::Abs>(make_shared<op::Abs>(make_shared<op::Abs>(innermost_abs))));

        run_passes(pass_manager, nested_sum_graph, {parm});
        auto sum = std::dynamic_pointer_cast<op::Sum>(innermost_abs->get_input_op(0));
        ASSERT_TRUE(sum);
        ASSERT_EQ(sum->get_reduction_axes(), axes);
        ASSERT_EQ(sum->get_input_op(0), parm);
    }
}

TEST(pattern, matcher)
{
    Shape shape{};
    auto a = make_shared<op::Parameter>(element::i32, shape);
    TestMatcher n(nullptr);
    ASSERT_TRUE(n.match(a, a));

    auto abs = make_shared<op::Abs>(a);
    auto any = std::make_shared<pattern::op::Any>(a);
    ASSERT_TRUE(n.match(any, abs));

    auto any_false =
        std::make_shared<pattern::op::Any>(a, [](std::shared_ptr<Node> no) { return false; });
    ASSERT_TRUE(n.match(any_false, a));

    auto pattern = std::make_shared<pattern::op::Label>(a);
    ASSERT_TRUE(n.match(pattern, a));
    ASSERT_EQ(n.get_pattern_map()[pattern], a);

    auto pattern_false =
        std::make_shared<pattern::op::Label>(a, [](std::shared_ptr<Node> no) { return false; });
    ASSERT_FALSE(n.match(pattern_false, a));

    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto d = make_shared<op::Parameter>(element::i32, shape);
    ASSERT_FALSE(n.match(d, b));

    ASSERT_FALSE(n.match(abs + b, b + b));
    ASSERT_TRUE(n.match(any + b, abs + b));

    ASSERT_TRUE(n.match(pattern + b, abs + b));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);

    ASSERT_TRUE(n.match(b + pattern, abs + b));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);

    auto c = make_shared<op::Parameter>(element::i32, shape);
    ASSERT_TRUE(n.match(c * (b + pattern), c * (abs + b)));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);

    ASSERT_TRUE(n.match(c * (any + b), c * (abs + b)));     //nested any
    ASSERT_TRUE(n.match(c * (any + b), (b + abs) * c));     //permutations w/ any
    ASSERT_TRUE(n.match(c * (any_false + b), c * (a + b))); //nested any
    ASSERT_TRUE(n.match(c * (any_false + b), (b + a) * c)); //permutations w/ any_false

    auto iconst1_0 = construct_constant_node(1);
    auto iconst1_1 = construct_constant_node(1);
    ASSERT_TRUE(n.match(pattern * iconst1_0, a * iconst1_1)); //different iconst
    ASSERT_EQ(n.get_pattern_map()[pattern], a);
    auto fconst1_0 = op::Constant::create(element::f32, shape, {1});
    auto patternf = std::make_shared<pattern::op::Label>(fconst1_0);
    ASSERT_TRUE(n.match(patternf * fconst1_0, a * iconst1_1)); //different iconst

    //Subgraph labels
    auto add = a + b;
    auto label = std::make_shared<pattern::op::Label>(add, nullptr, NodeVector{add});
    ASSERT_TRUE(n.match(label, add));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    ASSERT_FALSE(n.match(label, a - b));

    ASSERT_TRUE(n.match(make_shared<op::Abs>(label), make_shared<op::Abs>(add)));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    //Correlations
    auto label1 = std::make_shared<pattern::op::Label>(a);
    auto tmp = label1 + b;
    auto label2 = std::make_shared<pattern::op::Label>(tmp, nullptr, NodeVector{tmp});
    auto sub_label1 = label1 - label2;
    ASSERT_TRUE(n.match(sub_label1, a - add));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);

    ASSERT_FALSE(n.match(sub_label1, add - a));

    auto add_label1 = label1 + label2;
    ASSERT_TRUE(n.match(add_label1, add + a));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);
}

TEST(pattern, sum)
{
    //Sum
    TestMatcher n(nullptr);
    auto reducee_const = std::make_shared<op::Constant>(
        element::i32, Shape{2, 2}, std::vector<std::string>({"0", "0", "0", "0"}));
    auto sum_graph = xla_sum(reducee_const, AxisSet{0, 1});

    auto reduce_label = construct_sum_pattern();
    ASSERT_TRUE(n.match(reduce_label, sum_graph));
    ASSERT_EQ(n.get_pattern_map()[reduce_label], sum_graph);

    auto nested_sum_graph = make_shared<op::Abs>(make_shared<op::Abs>(
        make_shared<op::Abs>(make_shared<op::Abs>(make_shared<op::Abs>(sum_graph)))));

    auto nested_reduce_label = make_shared<op::Abs>(make_shared<op::Abs>(
        make_shared<op::Abs>(make_shared<op::Abs>(make_shared<op::Abs>(reduce_label)))));

    ASSERT_TRUE(n.match(nested_reduce_label, nested_sum_graph));
    ASSERT_EQ(n.get_pattern_map()[reduce_label], sum_graph);
}

TEST(pattern, mean)
{
    //construct mean
    TestMatcher n(nullptr);

    auto input = std::make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);

    auto mean_graph = construct_mean_graph();
    ASSERT_TRUE(n.match(mean_graph, mean));
    ASSERT_EQ(n.get_pattern_map()[mean_graph], mean);
}

TEST(pattern, variance)
{
    //construct variance
    TestMatcher n(nullptr);
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::Divide>(xmu, N);

    auto var_graph = construct_variance_graph();
    ASSERT_TRUE(n.match(var_graph, variance));
    ASSERT_EQ(n.get_pattern_map()[var_graph], variance);
}

TEST(pattern, previous_matches)
{
    using ngraph::pattern::Matcher;
    Shape shape{};
    Matcher::PatternMap previous_matches;
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto pattern = std::make_shared<pattern::op::Label>(b);
    auto abs = make_shared<op::Abs>(a);
    auto add = abs + b;
    {
        Matcher n(pattern + b);
        ASSERT_TRUE(n.match(add, previous_matches));
        ASSERT_EQ(n.get_pattern_map()[pattern], abs);
    }

    {
        Matcher n(pattern + b);
        previous_matches.insert(std::make_pair(pattern, a));
        ASSERT_FALSE(n.match(add, previous_matches));
    }
}

TEST(pattern, recurrent_pattern)
{
    using ngraph::pattern::Matcher;
    Shape shape{};
    ngraph::pattern::Matcher::PatternMap previous_matches;
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto rpattern = std::make_shared<pattern::op::Label>(b);
    auto iconst0 = construct_constant_node(0);
    auto abs = make_shared<op::Abs>(a);
    auto add1 = iconst0 + b;
    auto add2 = iconst0 + add1;
    auto add3 = iconst0 + add2;
    auto padd = iconst0 + rpattern;
    ngraph::pattern::RPatternMap matches;
    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    Matcher::match_recurring_pattern(add3, padd, rpattern, matches, empty_correlated_matches);
    ASSERT_EQ(matches.size(), 1);
    ASSERT_EQ(matches.count(rpattern), 1);
    auto recurrent_matches = matches[rpattern];
    ASSERT_EQ(recurrent_matches.at(0), add2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);

    //Multiple labels in a reccuring pattern
    auto iconst1 = construct_constant_node(1);
    auto iconst_label = std::make_shared<pattern::op::Label>(iconst1, nullptr, NodeVector{iconst1});
    auto add2_2 = iconst1 + add1;
    auto add3_2 = iconst0 + add2_2;
    auto padd2 = iconst_label + rpattern;
    matches.clear();
    Matcher::match_recurring_pattern(add3_2, padd2, rpattern, matches, empty_correlated_matches);
    ASSERT_EQ(matches.size(), 2);
    recurrent_matches = matches[rpattern];
    ASSERT_EQ(recurrent_matches.at(0), add2_2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);
    auto iconst_matches = matches[iconst_label];
    ASSERT_EQ(iconst_matches.at(0), iconst0);
    ASSERT_EQ(iconst_matches.at(1), iconst1);
    ASSERT_EQ(iconst_matches.at(2), iconst0);

    //Non-matching correlated labels
    matches.clear();
    std::set<std::shared_ptr<pattern::op::Label>> correlated_matches;
    correlated_matches.insert(iconst_label);
    Matcher::match_recurring_pattern(add3_2, padd2, rpattern, matches, correlated_matches);
    ASSERT_EQ(matches.size(), 2);
    iconst_matches = matches[iconst_label];
    ASSERT_EQ(iconst_matches.size(), 1);
    ASSERT_EQ(iconst_matches.at(0), iconst0);

    //Matching correlated labels
    matches.clear();
    Matcher::match_recurring_pattern(add3, padd2, rpattern, matches, correlated_matches);
    ASSERT_EQ(matches.size(), 2);
    recurrent_matches = matches[rpattern];
    ASSERT_EQ(recurrent_matches.at(0), add2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);
    iconst_matches = matches[iconst_label];
    ASSERT_EQ(iconst_matches.at(0), iconst0);
    ASSERT_EQ(iconst_matches.at(1), iconst0);
    ASSERT_EQ(iconst_matches.at(2), iconst0);
}

TEST(cpu_fusion, lstm)
{
    auto multiply_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Multiply>(n));
    };

    auto bias1 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto bias2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});

    auto param1_1 =
        std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100}, multiply_pred);
    auto param1_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});

    auto param2_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 50});
    auto param2_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 50});

    auto MatmulBias1 = std::make_shared<op::MatmulBias>(param1_1,
                                                        param1_2,
                                                        bias1,
                                                        param1_1->get_shape(),
                                                        param1_2->get_shape(),
                                                        false,
                                                        true,
                                                        AxisSet{0});
    auto MatmulBias2 = std::make_shared<op::MatmulBias>(param2_1,
                                                        param2_2,
                                                        bias2,
                                                        param2_1->get_shape(),
                                                        param2_2->get_shape(),
                                                        false,
                                                        true,
                                                        AxisSet{0});

    //auto X = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 400});

    auto X = std::make_shared<op::Add>(MatmulBias1, MatmulBias2);
    // construct forget gate
    auto input_slice_0 = std::make_shared<op::Slice>(X, Coordinate{0, 0}, Coordinate{10, 100});
    auto forget_gate = std::make_shared<op::Sigmoid>(input_slice_0);

    auto broadcast_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Broadcast>(n));
    };

    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto skip_ct_1 = std::make_shared<pattern::op::Any>(ct_1, broadcast_pred);
    // auto broadcast_ct_1 = std::make_shared<op::Broadcast>(ct_1, Shape{10, 100}, AxisSet{0, 1});
    // auto ct_1_label = std::make_shared<pattern::op::Label>(broadcast_ct_1, nullptr, NodeVector{broadcast_ct_1});
    auto multiply_forget_gate_ct_1 = std::make_shared<op::Multiply>(forget_gate, skip_ct_1);

    // construct input gate
    auto input_slice_1 = std::make_shared<op::Slice>(X, Coordinate{0, 100}, Coordinate{10, 200});
    auto input_gate = std::make_shared<op::Sigmoid>(input_slice_1);
    auto input_slice_2 = std::make_shared<op::Slice>(X, Coordinate{0, 200}, Coordinate{10, 300});
    auto tanh_1 = std::make_shared<op::Tanh>(input_slice_2);
    auto multiply_input_gate_tanh_1 = std::make_shared<op::Multiply>(input_gate, tanh_1);

    auto add_ct_1_input_gate_tanh_1 =
        std::make_shared<op::Add>(multiply_forget_gate_ct_1, multiply_input_gate_tanh_1);

    // construct output gate
    auto input_slice_3 = std::make_shared<op::Slice>(X, Coordinate{0, 300}, Coordinate{10, 400});
    auto output_gate = std::make_shared<op::Sigmoid>(input_slice_3);
    auto tanh_2 = std::make_shared<op::Tanh>(add_ct_1_input_gate_tanh_1);
    auto ht = std::make_shared<op::Multiply>(output_gate, tanh_2);

    pass::Manager pass_manager;
    //pass_manager.register_pass<pass::VisualizeTree>("bn_fprop_before_fusion.png");
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.register_pass<pass::VisualizeTree>("3LSTM_forward_after_cpu_fusion.json.pdf");
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/3LSTM_forward.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);

    NGRAPH_DEBUG << "RECURRENT MATCHING START!!!!!!";
    using ngraph::pattern::Matcher;
    ngraph::pattern::RPatternMap matches;
    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    Matcher::match_recurring_pattern(
        func->get_result()->get_input_op(0), ht, param1_1, matches, empty_correlated_matches);
    NGRAPH_DEBUG << "matches.size() " << matches.size();
}
