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
    , c_ref(get_user_inputs().user_constants.get_double("c_ref"))
    , diffusivity(get_user_inputs().user_constants.get_double("diffusivity"))
    , offset(get_user_inputs().user_constants.get_double("offset"))
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
    dealii::Point<dim> center(mesh_size / 2.0);
    double             rad  = 10.0;
    double             sdf  = ((point - center).norm_square() - rad * rad) / (2.0 * rad);
    double domain_parameter = 0.5 * ((1.0 + offset) - (1.0 - offset) * std::tanh(sdf));
    if (index == 0) // c
      {
        scalar_value = c0 * domain_parameter;
      }
    if (index == 1) // psi
      {
        scalar_value = domain_parameter;
      }
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int solve_block_id) const override

  {
    if (solve_block_id == 0) // c
      {
        // c terms
        ScalarValue c  = variable_list.template get_value<Scalar, OldOne>(0);
        ScalarGrad  cx = variable_list.template get_gradient<Scalar, OldOne>(0);

        // psi terms
        ScalarValue psi       = variable_list.template get_value<Scalar, Current>(1);
        ScalarGrad  psi_x     = variable_list.template get_gradient<Scalar, Current>(1);
        ScalarValue psi_x_mag = psi_x.norm() + offset;

        ScalarValue c_term1 = diffusivity * (psi_x * cx) / psi;
        ScalarValue c_term2 = (psi_x_mag / psi) * diffusivity * (-0.1 * (c - c_ref));

        ScalarGrad cx_term1 = -diffusivity * cx;

        ScalarValue eq_c  = c + sim_timer.get_timestep() * (c_term1 + c_term2);
        ScalarGrad  eqx_c = sim_timer.get_timestep() * cx_term1;

        variable_list.set_value_term(0, eq_c);
        variable_list.set_gradient_term(0, eqx_c);
      }
    else if (solve_block_id == 1) // pp
      {
        ScalarValue c   = variable_list.template get_value<Scalar, Current>(0);
        ScalarValue psi = variable_list.template get_value<Scalar, Current>(1);
        variable_list.set_value_term(2, c * psi);
      }
  }

  number c0;
  number c_ref;
  number offset;
  number diffusivity;
  // mutable std::uniform_real_distribution<number> dist;
};

PRISMS_PF_END_NAMESPACE
