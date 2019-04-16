#include <Omega_h_amr.hpp>
#include <Omega_h_amr_topology.hpp>
#include <Omega_h_amr_transfer.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_globals.hpp>
#include <Omega_h_hypercube.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_modify.hpp>
#include <Omega_h_unmap_mesh.hpp>

namespace Omega_h {

namespace amr {

static void refine_ghosted(Mesh* mesh) {
  Few<LOs, 4> mods2mds;
  for (Int mod_dim = 0; mod_dim <= mesh->dim(); ++mod_dim) {
    auto mds_are_mods = mesh->get_array<Byte>(mod_dim, "refine");
    mods2mds[mod_dim] = collect_marked(mds_are_mods);
  }
  for (Int prod_dim = 0; prod_dim <= mesh->dim(); ++prod_dim) {
    Few<LOs, 4> mods2nprods;
    Few<bool, 4> mods_have_prods;
    mods_have_prods[0] = false;
    for (Int mod_dim = 1; mod_dim <= mesh->dim(); ++mod_dim) {
      auto nprods_per_mods = hypercube_split_degree(mod_dim, prod_dim);
      mods2nprods[mod_dim] = LOs(mods2mds[mod_dim].size(), nprods_per_mods);
      mods_have_prods[mod_dim] = true;
    }
    auto rep2md_orders = get_rep2md_order(
        mesh, prod_dim, mods2mds, mods2nprods, mods_have_prods);
    auto name =
        std::string("rep_") + hypercube_singular_name(prod_dim) + "2md_order";
    for (Int mod_dim = prod_dim + 1; mod_dim <= mesh->dim(); ++mod_dim) {
      mesh->add_tag(mod_dim, name, 1, rep2md_orders[mod_dim]);
    }
  }
}

static void refine_elem_based(Mesh* mesh) {
  auto prod_counts = amr::count_refined(mesh);
  Few<Bytes, 4> mds_are_mods;
  Few<LOs, 4> mods2mds;
  Few<LOs, 4> mds2mods;
  for (Int mod_dim = 0; mod_dim <= mesh->dim(); ++mod_dim) {
    mds_are_mods[mod_dim] = mesh->get_array<Byte>(mod_dim, "refine");
    mods2mds[mod_dim] = collect_marked(mds_are_mods[mod_dim]);
    mds2mods[mod_dim] =
        invert_injective_map(mods2mds[mod_dim], mesh->nents(mod_dim));
  }
  auto new_mesh = mesh->copy_meta();
  Few<LOs, 4> mods2midverts;
  Few<LOs, 4> prods2new_ents;
  Few<LOs, 4> same_ents2old_ents;
  Few<LOs, 4> same_ents2new_ents;
  Few<LOs, 4> old_ents2new_ents;
  LOs old_lows2new_lows;
  for (Int prod_dim = 0; prod_dim <= mesh->dim(); ++prod_dim) {
    LOs prods2verts;
    if (prod_dim != VERT) {
      prods2verts =
          amr::get_refined_topology(mesh, prod_dim, prod_counts[prod_dim],
              mods2mds, mds2mods, mods2midverts, old_ents2new_ents[0]);
    }
    Few<LOs, 4> mods2prods;
    {
      LO offset = 0;
      for (Int mod_dim = max2(Int(EDGE), prod_dim); mod_dim <= mesh->dim();
           ++mod_dim) {
        auto nprods_per_mod = hypercube_split_degree(mod_dim, prod_dim);
        auto nmods_of_dim = mods2mds[mod_dim].size();
        mods2prods[mod_dim] =
            LOs(mods2mds[mod_dim].size() + 1, offset, nprods_per_mod);
        offset += nprods_per_mod * nmods_of_dim;
      }
    }
    modify_ents(mesh, &new_mesh, prod_dim, mods2mds, mds_are_mods, mods2prods,
        prods2verts, old_lows2new_lows, /*keep_mods*/ true,
        /*mods_can_be_shared*/ true, &(prods2new_ents[prod_dim]),
        &(same_ents2old_ents[prod_dim]), &(same_ents2new_ents[prod_dim]),
        &(old_ents2new_ents[prod_dim]));
    if (prod_dim == VERT) {
      mods2midverts[VERT] =
          unmap(mods2mds[VERT], old_ents2new_ents[prod_dim], 1);
      LO offset = 0;
      for (Int mod_dim = EDGE; mod_dim <= mesh->dim(); ++mod_dim) {
        OMEGA_H_CHECK(hypercube_split_degree(mod_dim, prod_dim) == 1);
        auto nmods_of_dim = mods2mds[mod_dim].size();
        auto begin = offset;
        auto end = begin + nmods_of_dim;
        mods2midverts[mod_dim] =
            unmap_range(begin, end, prods2new_ents[prod_dim], 1);
        offset = end;
      }
      amr::transfer_linear_interp(mesh, &new_mesh, mods2mds, mods2midverts,
          same_ents2old_ents[prod_dim], same_ents2new_ents[prod_dim]);
    }
    amr::transfer_levels(mesh, &new_mesh, prod_dim, mods2mds,
        prods2new_ents[prod_dim], same_ents2old_ents[prod_dim],
        same_ents2new_ents[prod_dim]);
    amr::transfer_leaves(mesh, &new_mesh, prod_dim, mods2mds,
        prods2new_ents[prod_dim], same_ents2old_ents[prod_dim],
        same_ents2new_ents[prod_dim], old_ents2new_ents[prod_dim]);
    old_lows2new_lows = old_ents2new_ents[prod_dim];
  }
  amr::transfer_parents(mesh, &new_mesh, mods2mds, prods2new_ents,
      same_ents2old_ents, same_ents2new_ents, old_ents2new_ents);
  amr::transfer_inherit(mesh, &new_mesh, prods2new_ents, same_ents2old_ents,
      same_ents2new_ents);
  *mesh = new_mesh;
}

void refine(Mesh* mesh, Bytes elems_are_marked) {
  OMEGA_H_CHECK(mesh->family() == OMEGA_H_HYPERCUBE);
  amr::mark_refined(mesh, elems_are_marked);
  mesh->set_parting(OMEGA_H_GHOSTED);
  amr::refine_ghosted(mesh);
  mesh->set_parting(OMEGA_H_ELEM_BASED);
  amr::refine_elem_based(mesh);
}

}  // namespace amr

}  // namespace Omega_h
