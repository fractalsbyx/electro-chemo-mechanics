// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <random>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , c0(get_user_inputs().user_constants.get_double("c0"))
  {}

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    const dealii::Tensor<1, dim> &mesh_size =
      get_user_inputs().spatial_discretization.rectangular_mesh.size;
    if (index == 0) // redundant
      {
        dealii::Point<dim> center(mesh_size / 2.0);
        double             rad = 5.0;
        double             sdf = (rad * rad - (point - center).norm_square()) / 2.0 * rad;
        scalar_value           = 0.5 * (1.0 + std::tanh(sdf));
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override

  {
    if (solve_block_id == 0) // c
      {
        /*
        ScalarValue c  = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  cx = variable_list.template get_gradient<Scalar, Current>(0);

        ScalarValue c_old = variable_list.template get_value<Scalar, OldOne>(0);

        ScalarValue r_c  = c - c_old;
        VectorValue r_cx = -1.0 * cx * sim_timer.get_timestep();

        variable_list.set_value_term(0, r_c);
        variable_list.set_gradient_term(0, r_cx);
        */
        ScalarValue c_old = variable_list.template get_value<Scalar, OldOne>(0);
        variable_list.set_value_term(0, c_old);
      }
    else if (solve_block_id == 1) // pp
      {
        variable_list.set_value_term(1, 0.0);
      }
  }

  void
  compute_lhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override

  {
    const double dt = sim_timer.get_timestep();
    if (solve_block_id == 0) // c
      {
        ScalarValue delta_c_val  = variable_list.template get_value<Scalar, LHS>(0);
        ScalarGrad  delta_c_grad = variable_list.template get_gradient<Scalar, LHS>(0);

        variable_list.set_value_term(0, delta_c_val);
        variable_list.set_gradient_term(0, 1.0 * dt * delta_c_grad);
      }
  }

  int                                            ic_type;
  number                                         c0;
  number                                         icamplitude;
  mutable std::uniform_real_distribution<number> dist;
};

PRISMS_PF_END_NAMESPACE
