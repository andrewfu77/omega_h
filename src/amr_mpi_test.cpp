#include <Omega_h_amr.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_print.hpp>
#include <iostream>

namespace Omega_h {

static Read<I32> codes_to_dim(Read<I8> codes) {
  Write<I32> dim(codes.size());
  auto functor = OMEGA_H_LAMBDA(LO i) {
    dim[i] = amr::code_parent_dim(codes[i]);
  };
  parallel_for(codes.size(), std::move(functor));
  return dim;
}

static Read<I32> codes_to_child(Read<I8> codes) {
  Write<I32> child(codes.size());
  auto functor = OMEGA_H_LAMBDA(LO i) {
    child[i] = amr::code_which_child(codes[i]);
  };
  parallel_for(codes.size(), std::move(functor));
  return child;
}

static void test_2D_case1(CommPtr comm) {
  auto mesh = build_box(comm, OMEGA_H_HYPERCUBE,
      2, 1., 0., 2, 1, 0); 
  vtk::FullWriter writer("out_amr_mpi_2D", &mesh);
  writer.write();
  {
    Write<Byte> marks(mesh.nelems(), 0);
    if (comm->rank() == 1) marks.set(0, 1);
    amr::refine(&mesh, marks);
  }

  if (comm->rank() == 0) {
    auto parents = mesh.ask_parents(1);
    std::cout << "parents idx: " << parents.parent_idx << std::endl;
    std::cout << "parents dim: " << codes_to_dim(parents.codes) << std::endl;
    std::cout << "parents which child: " << codes_to_child(parents.codes) << std::endl;
  }
 
  writer.write();
  {
    Write<Byte> marks(mesh.nelems(), 0); 
    if (comm->rank() == 1) marks.set(2, 1);
    amr::refine(&mesh, marks);
  }

  writer.write();
}

static void test_2D_case2(CommPtr comm) {
  auto mesh = build_box(comm, OMEGA_H_HYPERCUBE,
      2, 1., 0., 2, 1, 0); 
  { // refine element on rank 0
    Write<Byte> marks(mesh.nelems(), 0);
    if (comm->rank() == 0) marks.set(0, 1);
    const auto xfer_opts = TransferOpts();
    amr::refine(&mesh, marks, xfer_opts);
  }
  { // refine element on rank 1 (dies in get_amr_topology)
    Write<Byte> marks(mesh.nelems(), 0);
    if (comm->rank() == 1) marks.set(0, 1);
    const auto xfer_opts = TransferOpts();
    amr::refine(&mesh, marks, xfer_opts);
  }
}

}

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto comm = lib.world();
  OMEGA_H_CHECK(comm->size() == 2); 
  Omega_h::test_2D_case1(comm);
  Omega_h::test_2D_case2(comm);
}
