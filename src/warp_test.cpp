#include <iostream>

#include "Omega_h_array_ops.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_timer.hpp"
#include "access.hpp"
#include "internal.hpp"
#include "size.hpp"
#include "space.hpp"

using namespace Omega_h;

static void add_dye(Mesh* mesh) {
  auto dye_w = Write<Real>(mesh->nverts());
  auto coords = mesh->coords();
  auto dye_fun = LAMBDA(LO vert) {
    auto x = get_vector<3>(coords, vert);
    auto left_cen = vector_3(.25, .5, .5);
    auto right_cen = vector_3(.75, .5, .5);
    auto left_dist = norm(x - left_cen);
    auto right_dist = norm(x - right_cen);
    auto dist = min2(left_dist, right_dist);
    if (dist < .25) {
      auto dir = sign(left_dist - right_dist);
      dye_w[vert] = 4.0 * dir * (.25 - dist);
    } else {
      dye_w[vert] = 0;
    }
  };
  parallel_for(mesh->nverts(), dye_fun);
  mesh->add_tag(VERT, "dye", 1, Reals(dye_w));
}

static Reals form_pointwise(Mesh* mesh) {
  auto dim = mesh->dim();
  auto ecoords =
      average_field(mesh, dim, LOs(mesh->nelems(), 0, 1), dim, mesh->coords());
  auto pw_w = Write<Real>(mesh->nelems());
  auto pw_fun = LAMBDA(LO elem) { pw_w[elem] = ecoords[elem * dim]; };
  parallel_for(mesh->nelems(), pw_fun);
  return pw_w;
}

static void add_pointwise(Mesh* mesh) {
  auto data = form_pointwise(mesh);
  mesh->add_tag(mesh->dim(), "pointwise", 1, data);
}

static void postprocess_conserve(Mesh* mesh) {
  auto volume = measure_elements_real(mesh);
  auto mass = mesh->get_array<Real>(mesh->dim(), "mass");
  CHECK(are_close(1.0, get_sum(mesh->comm(), mass)));
  auto density = divide_each(mass, volume);
  mesh->add_tag(mesh->dim(), "density", 1, density);
}

static void postprocess_pointwise(Mesh* mesh) {
  auto data = mesh->get_array<Real>(mesh->dim(), "pointwise");
  auto expected = form_pointwise(mesh);
  auto diff = subtract_each(data, expected);
  mesh->add_tag(mesh->dim(), "pointwise_err", 1, diff);
}

int main(int argc, char** argv) {
  auto lib = Library(&argc, &argv);
  auto world = lib.world();
  constexpr Int dim = 3;
  Mesh mesh(&lib);
  if (world->rank() == 0) {
    auto nx = 10;
    build_box(&mesh, 1, 1, 1, nx, nx, (dim == 3) ? nx : 0);
    classify_by_angles(&mesh, PI / 4);
    mesh.reorder();
    mesh.reset_globals();
  }
  mesh.set_comm(world);
  mesh.balance();
  mesh.set_parting(OMEGA_H_GHOSTED);
  auto size = find_implied_size(&mesh);
  mesh.add_tag(VERT, "size", 1, size);
  add_dye(&mesh);
  mesh.add_tag(mesh.dim(), "mass", 1, measure_elements_real(&mesh));
  add_pointwise(&mesh);
  auto opts = AdaptOpts(&mesh);
  opts.xfer_opts.type_map["mass"] = OMEGA_H_CONSERVE;
  opts.xfer_opts.type_map["pointwise"] = OMEGA_H_POINTWISE;
  opts.xfer_opts.type_map["dye"] = OMEGA_H_LINEAR_INTERP;
  auto mid = zero_vector<dim>();
  mid[0] = mid[1] = .5;
  Now t0 = now();
  for (Int i = 0; i < 8; ++i) {
    auto coords = mesh.coords();
    Write<Real> warp_w(mesh.nverts() * dim);
    auto warp_fun = LAMBDA(LO vert) {
      auto x0 = get_vector<dim>(coords, vert);
      auto x1 = zero_vector<dim>();
      x1[0] = x0[0];
      x1[1] = x0[1];
      auto x2 = x1 - mid;
      auto polar_a = atan2(x2[1], x2[0]);
      auto polar_r = norm(x2);
      Real rot_a = 0;
      if (polar_r < 0.5) {
        rot_a = (PI / 8) * (2.0 * (0.5 - polar_r));
        if (i >= 4) rot_a = -rot_a;
      }
      auto dest_a = polar_a + rot_a;
      auto dst = x0;
      dst[0] = cos(dest_a) * polar_r;
      dst[1] = sin(dest_a) * polar_r;
      dst = dst + mid;
      auto w = dst - x0;
      set_vector<dim>(warp_w, vert, w);
    };
    parallel_for(mesh.nverts(), warp_fun);
    mesh.add_tag(VERT, "warp", dim, Reals(warp_w));
    while (warp_to_limit(&mesh, opts)) adapt(&mesh, opts);
  }
  Now t1 = now();
  mesh.set_parting(OMEGA_H_ELEM_BASED);
  if (mesh.comm()->rank() == 0) {
    std::cout << "test took " << (t1 - t0) << " seconds\n";
  }
  postprocess_conserve(&mesh);
  postprocess_pointwise(&mesh);
  bool ok = check_regression("gold_warp", &mesh);
  if (!ok) return 2;
  return 0;
}
