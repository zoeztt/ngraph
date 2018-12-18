.. allreduce.rst:

#########
AllReduce
#########

.. code-block:: cpp

   AllReduce // Collective operation


Description
===========

Combines values from all processes or devices and distributes the result back
to all processes or devices.


Inputs
------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``arg``         | ``f32``        | Any                            |
|                 | ``f64``        |                                |
+-----------------+-------------------------+--------------------------------+


Outputs
-------

+-----------------+-------------------------+--------------------------------+
| Name            | Element Type            | Shape                          |
+=================+=========================+================================+
| ``output``      | ``f32``        | Same as ``arg``                |
|                 | ``f64``        |                                |
+-----------------+-------------------------+--------------------------------+


C++ Interface
=============

.. doxygenclass:: ngraph::op::AllReduce
   :project: ngraph
   :members:
