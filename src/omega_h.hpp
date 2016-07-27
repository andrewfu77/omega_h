#ifndef OMEGA_H_HPP
#define OMEGA_H_HPP

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "omega_h_c.h"
#include "omega_h_kokkos.hpp"

namespace osh {

typedef std::int8_t I8;
typedef std::int16_t I16;
typedef std::int32_t I32;
typedef std::int64_t I64;
typedef I32 Int;
typedef I32 LO;
typedef I64 GO;
typedef double Real;

constexpr Real PI = 3.141592653589793;

template <typename T>
class HostWrite;

template <typename T>
class Write {
#ifdef OSH_USE_KOKKOS
  Kokkos::View<T*> view_;
#else
  std::shared_ptr<T> ptr_;
  LO size_;
#endif
  /* separate boolean instead of data()==nullptr,
     see Kokkos issue #244 */
  bool exists_;

 public:
  OSH_INLINE Write();
#ifdef OSH_USE_KOKKOS
  Write(Kokkos::View<T*> view);
#endif
  Write(LO size);
  Write(LO size, T value);
  Write(LO size, T offset, T stride);
  Write(HostWrite<T> host_write);
  LO size() const;
  OSH_DEVICE T& operator[](LO i) const {
#ifdef OSH_CHECK_BOUNDS
    OSH_CHECK(0 <= i);
    OSH_CHECK(i < size());
#endif
#ifdef OSH_USE_KOKKOS
    return view_(i);
#else
    return ptr_.get()[i];
#endif
  }
  T* data() const;
#ifdef OSH_USE_KOKKOS
  Kokkos::View<T*> view() const;
#endif
  void set(LO i, T value) const;
  T get(LO i) const;
  bool exists() const;
};

template <typename T>
OSH_INLINE Write<T>::Write()
    :
#ifdef OSH_USE_KOKKOS
      view_()
#else
      ptr_(),
      size_(0)
#endif
      ,
      exists_(false) {
}

template <typename T>
class Read {
  Write<T> write_;

 public:
  OSH_INLINE Read() {}
  Read(Write<T> write);
  Read(LO size, T value);
  Read(LO size, T offset, T stride);
  Read(std::initializer_list<T> l);
  LO size() const;
  OSH_DEVICE T const& operator[](LO i) const { return write_[i]; }
  T const* data() const;
#ifdef OSH_USE_KOKKOS
  Kokkos::View<const T*> view() const;
#endif
  T get(LO i) const;
  T last() const;
  bool exists() const;
};

class LOs : public Read<LO> {
 public:
  OSH_INLINE LOs() {}
  OSH_INLINE LOs(Read<LO> base) : Read<LO>(base) {}
  LOs(Write<LO> write);
  LOs(LO size, LO value);
  LOs(LO size, LO offset, LO stride);
  LOs(std::initializer_list<LO> l);
};

template <typename T>
Read<T> permute(Read<T> a_data, LOs a2b, Int width);

class Reals : public Read<Real> {
 public:
  Reals();
  OSH_INLINE Reals(Read<Real> base) : Read<Real>(base) {}
  Reals(Write<Real> write);
  Reals(LO size, Real value);
  Reals(std::initializer_list<Real> l);
};

template <typename T>
class HostRead {
  Read<T> read_;
#ifdef OSH_USE_KOKKOS
  typename Kokkos::View<const T*>::HostMirror mirror_;
#endif
 public:
  HostRead();
  HostRead(Read<T> read);
  LO size() const;
  inline T const& operator[](LO i) const {
#ifdef OSH_USE_KOKKOS
#ifdef OSH_CHECK_BOUNDS
    OSH_CHECK(0 <= i);
    OSH_CHECK(i < size());
#endif
    return mirror_(i);
#else
    return read_[i];
#endif
  }
  T const* data() const;
  T last() const;
};

template <typename T>
class HostWrite {
  Write<T> write_;
#ifdef OSH_USE_KOKKOS
  typename Kokkos::View<T*>::HostMirror mirror_;
#endif
 public:
  HostWrite();
  HostWrite(LO size);
  HostWrite(LO size, T offset, T stride);
  HostWrite(Write<T> write);
  HostWrite(std::initializer_list<T> l);
  Write<T> write() const;
  LO size() const;
  inline T& operator[](LO i) const {
#ifdef OSH_USE_KOKKOS
#ifdef OSH_CHECK_BOUNDS
    OSH_CHECK(0 <= i);
    OSH_CHECK(i < size());
#endif
    return mirror_(i);
#else
    return write_[i];
#endif
  }
  T* data() const;
};

class TagBase {
 public:
  TagBase(std::string const& name, Int ncomps, osh_xfer xfer);
  virtual ~TagBase();
  std::string const& name() const;
  Int ncomps() const;
  osh_xfer xfer() const;
  virtual osh_type type() const = 0;

 private:
  std::string name_;
  Int ncomps_;
  osh_xfer xfer_;
};

template <typename T>
class Tag : public TagBase {
 public:
  Tag(std::string const& name, Int ncomps, osh_xfer xfer);
  Read<T> array() const;
  void set_array(Read<T> array);
  virtual osh_type type() const override;

 private:
  Read<T> array_;
};

struct Remotes {
  Remotes() {}
  Remotes(Read<I32> ranks_, LOs idxs_) : ranks(ranks_), idxs(idxs_) {}
  Read<I32> ranks;
  LOs idxs;
};

struct Int128 {
  std::int64_t high;
  std::uint64_t low;
  OSH_INLINE Int128();
  OSH_INLINE Int128(std::int64_t h, std::uint64_t l);
  OSH_INLINE Int128(std::int64_t value);
  OSH_INLINE void operator=(Int128 const& rhs) volatile;
  OSH_INLINE Int128(Int128 const& rhs);
  OSH_INLINE Int128(const volatile Int128& rhs);
  double to_double(double unit) const;
  void print(std::ostream& o) const;
  static OSH_INLINE Int128 from_double(double value, double unit);
};

class Comm;

typedef std::shared_ptr<Comm> CommPtr;

class Comm {
#ifdef OSH_USE_MPI
  MPI_Comm impl_;
#endif
  Read<I32> srcs_;
  HostRead<I32> host_srcs_;
  Read<I32> dsts_;
  HostRead<I32> host_dsts_;

 public:
  Comm();
#ifdef OSH_USE_MPI
  Comm(MPI_Comm impl);
#else
  Comm(bool is_graph, bool sends_to_self);
#endif
  ~Comm();
  static CommPtr world();
  static CommPtr self();
  I32 rank() const;
  I32 size() const;
  CommPtr dup() const;
  CommPtr split(I32 color, I32 key) const;
  CommPtr graph(Read<I32> dsts) const;
  CommPtr graph_adjacent(Read<I32> srcs, Read<I32> dsts) const;
  CommPtr graph_inverse() const;
  Read<I32> sources() const;
  Read<I32> destinations() const;
  template <typename T>
  T allreduce(T x, osh_op op) const;
  bool reduce_or(bool x) const;
  bool reduce_and(bool x) const;
  Int128 add_int128(Int128 x) const;
  template <typename T>
  T exscan(T x, osh_op op) const;
  template <typename T>
  void bcast(T& x) const;
  void bcast_string(std::string& s) const;
  template <typename T>
  Read<T> allgather(T x) const;
  template <typename T>
  Read<T> alltoall(Read<T> x) const;
  template <typename T>
  Read<T> alltoallv(Read<T> sendbuf, Read<LO> sendcounts, Read<LO> sdispls,
                    Read<LO> recvcounts, Read<LO> rdispls) const;
  void barrier() const;
};

class Dist {
  CommPtr parent_comm_;
  LOs roots2items_[2];
  LOs items2content_[2];
  LOs msgs2content_[2];
  CommPtr comm_[2];

 public:
  Dist();
  Dist(Dist const& other);
  Dist& operator=(Dist const& other);
  Dist(CommPtr comm, Remotes fitems2rroots, LO nrroots);
  void set_parent_comm(CommPtr parent_comm);
  void set_dest_ranks(Read<I32> items2ranks);
  void set_dest_idxs(LOs fitems2rroots, LO nrroots);
  void set_roots2items(LOs froots2fitems);
  Dist invert() const;
  template <typename T>
  Read<T> exch(Read<T> data, Int width) const;
  template <typename T>
  Read<T> exch_reduce(Read<T> data, Int width, osh_op op) const;
  CommPtr parent_comm() const;
  CommPtr comm() const;
  LOs msgs2content() const;
  LOs content2msgs() const;
  LOs items2msgs() const;
  LOs roots2items() const;
  Read<I32> msgs2ranks() const;
  Read<I32> items2ranks() const;
  LOs items2dest_idxs() const;
  Remotes items2dests() const;
  LO nitems() const;
  LO nroots() const;
  LO ndests() const;
  LO nsrcs() const;
  void change_comm(CommPtr new_comm);
  Remotes exch(Remotes data, Int width) const;

 private:
  void copy(Dist const& other);
  enum { F, R };
};

struct Graph {
  Graph() {}
  explicit Graph(LOs ab2b_) : ab2b(ab2b_) {}
  Graph(LOs a2ab_, LOs ab2b_) : a2ab(a2ab_), ab2b(ab2b_) {}
  LOs a2ab;
  LOs ab2b;
};

enum { DIMS = OSH_DIMS };

enum { VERT = OSH_VERT, EDGE = OSH_EDGE, TRI = OSH_TRI, TET = OSH_TET };

struct Adj : public Graph {
  Adj() {}
  explicit Adj(LOs ab2b_) : Graph(ab2b_) {}
  Adj(LOs ab2b_, Read<I8> codes_) : Graph(ab2b_), codes(codes_) {}
  Adj(LOs a2ab_, LOs ab2b_, Read<I8> codes_)
      : Graph(a2ab_, ab2b_), codes(codes_) {}
  Adj(LOs a2ab_, LOs ab2b_) : Graph(a2ab_, ab2b_) {}
  Adj(Graph g) : Graph(g) {}
  Read<I8> codes;
};

void find_matches(Int dim, LOs av2v, LOs bv2v, Adj v2b, LOs* a2b_out,
                  Read<I8>* codes_out);

class Library {
 public:
  Library(Library const&) {}
  inline Library(int* argc, char*** argv) { osh_init(argc, argv); }
  ~Library();
  CommPtr world() const;
  CommPtr self() const;
};

namespace inertia {
struct Rib;
}

class Mesh {
 public:
  Mesh();
  void set_comm(CommPtr comm);
  void set_dim(Int dim);
  void set_verts(LO nverts);
  void set_ents(Int dim, Adj down);
  void keep_canonical_globals(bool yn);
  CommPtr comm() const;
  osh_parting parting() const;
  Int dim() const;
  LO nents(Int dim) const;
  LO nelems() const;
  LO ntets() const;
  LO ntris() const;
  LO nedges() const;
  LO nverts() const;
  GO nglobal_ents(Int dim);
  template <typename T>
  void add_tag(Int dim, std::string const& name, Int ncomps, osh_xfer xfer);
  template <typename T>
  void add_tag(Int dim, std::string const& name, Int ncomps, osh_xfer xfer,
               Read<T> array);
  template <typename T>
  void set_tag(Int dim, std::string const& name, Read<T> array);
  TagBase const* get_tagbase(Int dim, std::string const& name) const;
  template <typename T>
  Tag<T> const* get_tag(Int dim, std::string const& name) const;
  template <typename T>
  Read<T> get_array(Int dim, std::string const& name) const;
  void remove_tag(Int dim, std::string const& name);
  bool has_tag(Int dim, std::string const& name) const;
  Int ntags(Int dim) const;
  TagBase const* get_tag(Int dim, Int i) const;
  bool has_ents(Int dim) const;
  bool has_adj(Int from, Int to) const;
  Adj get_adj(Int from, Int to) const;
  Adj ask_down(Int from, Int to);
  LOs ask_verts_of(Int dim);
  Adj ask_up(Int from, Int to);
  Graph ask_star(Int dim);
  Graph ask_dual();

 public:
  typedef std::shared_ptr<TagBase> TagPtr;
  typedef std::shared_ptr<Adj> AdjPtr;
  typedef std::shared_ptr<Dist> DistPtr;
  typedef std::shared_ptr<inertia::Rib> RibPtr;

 private:
  typedef std::vector<TagPtr> TagVector;
  typedef TagVector::iterator TagIter;
  typedef TagVector::const_iterator TagCIter;
  TagIter tag_iter(Int dim, std::string const& name);
  TagCIter tag_iter(Int dim, std::string const& name) const;
  void check_dim(Int dim) const;
  void check_dim2(Int dim) const;
  void add_adj(Int from, Int to, Adj adj);
  Adj derive_adj(Int from, Int to);
  Adj ask_adj(Int from, Int to);
  void react_to_set_tag(Int dim, std::string const& name);
  Int dim_;
  CommPtr comm_;
  Int parting_;
  LO nents_[DIMS];
  TagVector tags_[DIMS];
  AdjPtr adjs_[DIMS][DIMS];
  Remotes owners_[DIMS];
  DistPtr dists_[DIMS];
  RibPtr rib_hints_;
  bool keeps_canonical_globals_;

 public:
  void add_coords(Reals array);
  Reals coords() const;
  void set_coords(Reals array);
  Read<GO> ask_globals(Int dim);
  void reset_globals();
  Reals ask_lengths();
  Reals ask_qualities();
  void set_owners(Int dim, Remotes owners);
  Remotes ask_owners(Int dim);
  Read<I8> owned(Int dim);
  Dist ask_dist(Int dim);
  void set_parting(osh_parting parting, bool verbose = false);
  void migrate(Remotes new_elems2old_owners, bool verbose = false);
  void reorder();
  void balance();
  Graph ask_graph(Int from, Int to);
  template <typename T>
  Read<T> sync_array(Int ent_dim, Read<T> a, Int width);
  template <typename T>
  Read<T> sync_subset_array(Int ent_dim, Read<T> a_data, LOs a2e, T default_val,
                            Int width);
  template <typename T>
  Read<T> reduce_array(Int ent_dim, Read<T> a, Int width, osh_op op);
  void sync_tag(Int dim, std::string const& name);
  void reduce_tag(Int dim, std::string const& name, osh_op op);
  bool operator==(Mesh& other);
  Real min_quality();
  bool could_be_shared(Int ent_dim) const;
  bool owners_have_all_upward(Int ent_dim) const;
  Mesh copy_meta() const;
  bool keeps_canonical_globals() const;
  RibPtr rib_hints() const;
  void set_rib_hints(RibPtr hints);
};

namespace gmsh {
void read(std::istream& stream, Library const& lib, Mesh* mesh);
void read(std::string const& filename, Library const& lib, Mesh* mesh);
}

namespace vtk {
void write_vtu(std::ostream& stream, Mesh* mesh, Int cell_dim);
void write_vtu(std::string const& filename, Mesh* mesh, Int cell_dim);
void write_parallel(std::string const& path, Mesh* mesh, Int cell_dim);
class Writer {
  Mesh* mesh_;
  std::string root_path_;
  Int cell_dim_;
  Int step_;
  std::streampos pvd_pos_;

 public:
  Writer();
  Writer(Mesh* mesh, std::string const& root_path, Int cell_dim);
  Writer(Writer const& other);
  ~Writer();
  void write(Real time);
  void write();
};
class FullWriter {
  std::vector<Writer> writers_;

 public:
  FullWriter(Mesh* mesh, std::string const& root_path);
  ~FullWriter();
  void write(Real time);
  void write();
};
}  // end namespace vtk

/* returns true if the mesh was modified. */
bool adapt(Mesh* mesh, Real qual_floor, Real qual_ceil, Real len_floor,
           Real len_ceil, Int nlayers, Int verbosity);

namespace binary {
void write(std::string const& path, Mesh* mesh);
void read(std::string const& path, CommPtr comm, Mesh* mesh);
}

osh_comparison compare_meshes(Mesh* a, Mesh* b, Real tol, Real floor,
                              bool verbose, bool full = true);
bool check_regression(std::string const& prefix, Mesh* mesh, Real tol,
                      Real floor);

void build_from_elems2verts(Mesh* mesh, CommPtr comm, Int edim, LOs ev2v,
                            Read<GO> vert_globals);
void build_from_elems2verts(Mesh* mesh, Library const& lib, Int edim, LOs ev2v,
                            LO nverts);
void build_from_elems_and_coords(Mesh* mesh, Library const& lib, Int edim,
                                 LOs ev2v, Reals coords);
void build_box(Mesh* mesh, Library const& lib, Real x, Real y, Real z, LO nx,
               LO ny, LO nz);

Real repro_sum(Reals a);
Real repro_sum(CommPtr comm, Reals a);
void repro_sum(CommPtr comm, Reals a, Int ncomps, Real result[]);

OSH_INLINE bool code_is_flipped(I8 code) { return code & 1; }

OSH_INLINE Int code_rotation(I8 code) { return (code >> 1) & 3; }

OSH_INLINE Int code_which_down(I8 code) { return (code >> 3); }

Read<I8> mark_class_closure(Mesh* mesh, Int ent_dim, Int class_dim,
                            I32 class_id);
Read<I8> mark_class_closures(Mesh* mesh, Int ent_dim,
                             std::vector<Int> class_dims,
                             std::vector<I32> class_ids);

LOs collect_marked(Read<I8> marks);

bool warp_to_limit(Mesh* mesh, Real min_qual);
bool approach_metric(Mesh* mesh, Real min_qual);

Reals find_identity_size(Mesh* mesh);
Reals find_identity_metric(Mesh* mesh);

void classify_by_angles(Mesh* mesh, Real sharp_angle);

template <typename T, Int n>
class Few {
  T array_[n];

 public:
  enum { size = n };
  OSH_INLINE T& operator[](Int i) { return array_[i]; }
  OSH_INLINE T const& operator[](Int i) const { return array_[i]; }
  OSH_INLINE Few() {}
  Few(std::initializer_list<T> l) {
    Int i = 0;
    for (auto v : l) array_[i++] = v;
  }
  OSH_INLINE void operator=(Few<T, n> const& rhs) volatile {
    for (Int i = 0; i < n; ++i) array_[i] = rhs.array_[i];
  }
  OSH_INLINE Few(Few<T, n> const& rhs) {
    for (Int i = 0; i < n; ++i) array_[i] = rhs.array_[i];
  }
  OSH_INLINE Few(const volatile Few<T, n>& rhs) {
    for (Int i = 0; i < n; ++i) array_[i] = rhs.array_[i];
  }
};

OSH_INLINE Real square(Real x) { return x * x; }

template <Int n>
class Vector : public Few<Real, n> {
 public:
  OSH_INLINE Vector() {}
  inline Vector(std::initializer_list<Real> l) : Few<Real, n>(l) {}
  OSH_INLINE void operator=(Vector<n> const& rhs) volatile {
    Few<Real, n>::operator=(rhs);
  }
  OSH_INLINE Vector(Vector<n> const& rhs) : Few<Real, n>(rhs) {}
  OSH_INLINE Vector(const volatile Vector<n>& rhs) : Few<Real, n>(rhs) {}
};

template <Int n>
OSH_INLINE Vector<n> operator+(Vector<n> a, Vector<n> b) {
  Vector<n> c;
  for (Int i = 0; i < n; ++i) c[i] = a[i] + b[i];
  return c;
}

template <Int n>
OSH_INLINE Vector<n> operator-(Vector<n> a, Vector<n> b) {
  Vector<n> c;
  for (Int i = 0; i < n; ++i) c[i] = a[i] - b[i];
  return c;
}

template <Int n>
OSH_INLINE Vector<n> operator*(Vector<n> a, Real b) {
  Vector<n> c;
  for (Int i = 0; i < n; ++i) c[i] = a[i] * b;
  return c;
}

template <Int n>
OSH_INLINE Vector<n> operator*(Real a, Vector<n> b) {
  return b * a;
}

OSH_INLINE Vector<2> vector_2(Real x, Real y) {
  Vector<2> v;
  v[0] = x;
  v[1] = y;
  return v;
}

OSH_INLINE Vector<3> vector_3(Real x, Real y, Real z) {
  Vector<3> v;
  v[0] = x;
  v[1] = y;
  v[2] = z;
  return v;
}

/* column-first storage and indexing !!! */
template <Int m, Int n>
class Matrix : public Few<Vector<m>, n> {
 public:
  OSH_INLINE Matrix() {}
  /* these constructors accept the matrix in
   * row-first order for convenience.
   */
  inline Matrix(std::initializer_list<Vector<m>> l) : Few<Vector<m>, n>(l) {}
  inline Matrix(std::initializer_list<Real> l);
  OSH_INLINE void operator=(Matrix<m, n> const& rhs) volatile {
    Few<Vector<m>, n>::operator=(rhs);
  }
  OSH_INLINE Matrix(Matrix<m, n> const& rhs) : Few<Vector<m>, n>(rhs) {}
  OSH_INLINE Matrix(const volatile Matrix<m, n>& rhs)
      : Few<Vector<m>, n>(rhs) {}
};

template <Int m, Int n>
OSH_INLINE Vector<m> operator*(Matrix<m, n> a, Vector<n> b) {
  Vector<m> c = a[0] * b[0];
  for (Int j = 1; j < n; ++j) c = c + a[j] * b[j];
  return c;
}

template <Int m, Int n, Int p>
OSH_INLINE Matrix<m, n> operator*(Matrix<m, p> a, Matrix<p, n> b) {
  Matrix<m, n> c;
  for (Int j = 0; j < n; ++j) c[j] = a * b[j];
  return c;
}

template <Int m, Int n>
OSH_INLINE Matrix<n, m> transpose(Matrix<m, n> a) {
  Matrix<n, m> b;
  for (Int i = 0; i < m; ++i)
    for (Int j = 0; j < n; ++j) b[i][j] = a[j][i];
  return b;
}

template <Int m, Int n>
OSH_INLINE Matrix<m, n> identity_matrix() {
  Matrix<m, n> a;
  for (Int j = 0; j < n; ++j)
    for (Int i = 0; i < m; ++i) a[j][i] = (i == j);
  return a;
}

template <Int m>
OSH_INLINE Matrix<m, m> diagonal(Vector<m> v) {
  Matrix<m, m> a;
  for (Int i = 0; i < m; ++i)
    for (Int j = i + 1; j < m; ++j) a[i][j] = a[j][i] = 0.0;
  for (Int i = 0; i < m; ++i) a[i][i] = v[i];
  return a;
}

template <Int n>
OSH_DEVICE void set_vector(Write<Real> const& a, Int i, Vector<n> v) {
  for (Int j = 0; j < n; ++j) a[i * n + j] = v[j];
}

OSH_INLINE constexpr Int symm_dofs(Int dim) { return ((dim + 1) * dim) / 2; }

OSH_INLINE Vector<3> symm2vector(Matrix<2, 2> symm) {
  Vector<3> v;
  v[0] = symm[0][0];
  v[1] = symm[1][1];
  v[2] = symm[1][0];
  return v;
}

OSH_INLINE Vector<6> symm2vector(Matrix<3, 3> symm) {
  Vector<6> v;
  v[0] = symm[0][0];
  v[1] = symm[1][1];
  v[2] = symm[2][2];
  v[3] = symm[1][0];
  v[4] = symm[2][1];
  v[5] = symm[2][0];
  return v;
}

template <Int n>
OSH_DEVICE void set_symm(Write<Real> const& a, Int i, Matrix<n, n> symm) {
  set_vector(a, i, symm2vector(symm));
}

template <Int dim>
OSH_INLINE Vector<dim> metric_eigenvalues(Vector<dim> h) {
  Vector<dim> l;
  for (Int i = 0; i < dim; ++i) l[i] = 1.0 / square(h[i]);
  return l;
}

template <Int dim>
OSH_INLINE Matrix<dim, dim> compose_metric(Matrix<dim, dim> r, Vector<dim> h) {
  auto l = metric_eigenvalues(h);
  return r * diagonal(l) * transpose(r);
}

template <Int dim>
Reals repeat_symm(LO n, Matrix<dim, dim> symm);
extern template Reals repeat_symm(LO n, Matrix<3, 3> symm);
extern template Reals repeat_symm(LO n, Matrix<2, 2> symm);

/* begin explicit instantiation declarations */
#define OSH_EXPL_INST_DECL(T)                                                  \
  extern template class Read<T>;                                               \
  extern template class Write<T>;                                              \
  extern template class HostRead<T>;                                           \
  extern template class HostWrite<T>;                                          \
  extern template T Comm::allreduce(T x, osh_op op) const;                     \
  extern template T Comm::exscan(T x, osh_op op) const;                        \
  extern template void Comm::bcast(T& x) const;                                \
  extern template Read<T> Comm::allgather(T x) const;                          \
  extern template Read<T> Comm::alltoall(Read<T> x) const;                     \
  extern template Read<T> Comm::alltoallv(                                     \
      Read<T> sendbuf, Read<LO> sendcounts, Read<LO> sdispls,                  \
      Read<LO> recvcounts, Read<LO> rdispls) const;                            \
  extern template Read<T> Dist::exch(Read<T> data, Int width) const;           \
  extern template Read<T> Dist::exch_reduce<T>(Read<T> data, Int width,        \
                                               osh_op op) const;               \
  extern template Tag<T> const* Mesh::get_tag<T>(                              \
      Int dim, std::string const& name) const;                                 \
  extern template Read<T> Mesh::get_array<T>(Int dim, std::string const& name) \
      const;                                                                   \
  extern template void Mesh::add_tag<T>(Int dim, std::string const& name,      \
                                        Int ncomps, osh_xfer xfer);            \
  extern template void Mesh::add_tag<T>(Int dim, std::string const& name,      \
                                        Int ncomps, osh_xfer xfer,             \
                                        Read<T> array);                        \
  extern template void Mesh::set_tag(Int dim, std::string const& name,         \
                                     Read<T> array);                           \
  extern template Read<T> Mesh::sync_array(Int ent_dim, Read<T> a, Int width); \
  extern template Read<T> Mesh::sync_subset_array(                             \
      Int ent_dim, Read<T> a_data, LOs a2e, T default_val, Int width);         \
  extern template Read<T> Mesh::reduce_array(Int ent_dim, Read<T> a,           \
                                             Int width, osh_op op);
OSH_EXPL_INST_DECL(I8)
OSH_EXPL_INST_DECL(I32)
OSH_EXPL_INST_DECL(I64)
OSH_EXPL_INST_DECL(Real)
#undef OSH_EXPL_INST_DECL
/* end explicit instantiation declarations */

}  // end namespace osh

#endif