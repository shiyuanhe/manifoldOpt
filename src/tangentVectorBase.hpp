#ifndef TANGENT_VECTORBASE_CLASS
#define TANGENT_VECTORBASE_CLASS

#include "manifold_include.hpp"

class tangent_vector_base {
public:
  void resize(size_type);
  void assign_names(const vector<string> &);
  // tangent matrices are indexed by 0,1,2,3...
  arma::mat &operator()(const size_type int_index = 0);
  // tangent matrices are indexed by strings, eg. U, V, Sigma,...
  // These string are specified by the user
  const arma::mat &operator()(const size_type int_index = 0) const;
  arma::mat &operator()(const string str_index);
  const arma::mat &operator()(const string str_index) const;

  auto operator+(const tangent_vector_base &) const -> tangent_vector_base;
  auto operator-(const tangent_vector_base &) const -> tangent_vector_base;
  auto operator-() const -> tangent_vector_base;
  auto operator*(const double scalar) const -> tangent_vector_base;

private:
  size_type size;
  // vector<tangent_vector_base> tangent_components;
  vector<arma::mat> tangent_matrices;
  map<string, size_type> name_mapping;

  void initial_copy(const tangent_vector_base &);
  // void operation_check(const tangent_vector_base &) const;
  void index_check(const size_type int_index) const;
  auto index_check(const string str_index) const -> size_type;
};

#endif
