#include "tangentVectorBase.hpp"

auto tangent_vector_base::operator+(const tangent_vector_base &rhs) const
    -> tangent_vector_base {

  tangent_vector_base res;
  res.initial_copy(*this);
  for (size_type i = 0; i < size; i++)
    res.tangent_matrices.at(i) =
        tangent_matrices.at(i) + rhs.tangent_matrices.at(i);

  return res;
}

auto tangent_vector_base::operator-(const tangent_vector_base &rhs) const
    -> tangent_vector_base {

  tangent_vector_base res;
  res.initial_copy(*this);
  for (size_type i = 0; i < size; i++)
    res.tangent_matrices.at(i) =
        tangent_matrices.at(i) - rhs.tangent_matrices.at(i);

  return res;
}

auto tangent_vector_base::operator-() const -> tangent_vector_base {

  tangent_vector_base res;
  res.initial_copy(*this);
  for (size_type i = 0; i < size; i++)
    res.tangent_matrices.at(i) = -tangent_matrices.at(i);

  return res;
}

auto tangent_vector_base::operator*(const double scalar) const
    -> tangent_vector_base {

  tangent_vector_base res;
  res.initial_copy(*this);
  for (size_type i = 0; i < size; i++)
    res.tangent_matrices.at(i) = tangent_matrices.at(i) * scalar;

  return res;
}

// void tangent_vector_base::operation_check(
//     const tangent_vector_base &rhs) const {
//   if (id_code.length() == 0 || size == 0)
//     throw runtime_error("Empty tangent vector!");
//   if (id_code != rhs.id_code || size != rhs.size)
//     throw runtime_error("Two tangent vectors not in same vector space!");
// }

arma::mat &tangent_vector_base::operator()(const size_type int_index) {
  index_check(int_index);
  return tangent_matrices[int_index];
}

const arma::mat &tangent_vector_base::
operator()(const size_type int_index) const {
  index_check(int_index);
  return tangent_matrices[int_index];
}

void tangent_vector_base::index_check(const size_type int_index) const {
  if (int_index >= size)
    throw range_error("Tangent index out of bound!");
}

arma::mat &tangent_vector_base::operator()(const string str_index) {
  size_type int_index;

  int_index = index_check(str_index);

  return tangent_matrices[int_index];
}

const arma::mat &tangent_vector_base::operator()(const string str_index) const {
  size_type int_index;
  int_index = index_check(str_index);
  return tangent_matrices[int_index];
}

size_type tangent_vector_base::index_check(const string str_index) const {
  size_type int_index = 0;
  map<string, size_type>::const_iterator it;

  it = name_mapping.find(str_index);
  if (it == name_mapping.end())
    throw range_error("Tangent component name out of bound!");
  int_index = it->second;
  return int_index;
}

void tangent_vector_base::initial_copy(const tangent_vector_base &tv) {
  resize(tv.size);
  name_mapping = tv.name_mapping;
}

void tangent_vector_base::resize(size_type new_size) {
  size = new_size;
  name_mapping.clear();
  tangent_matrices.clear();
  tangent_matrices.resize(size);
}

void tangent_vector_base::assign_names(const vector<string> &component_names) {
  if (component_names.size() != size)
    throw runtime_error("Incorrect Component Names Length~");

  std::pair<string, size_type> insert_pair;
  for (size_type i = 0; i < size; i++) {
    insert_pair.first = component_names[i];
    insert_pair.second = i;
    name_mapping.insert(insert_pair);
  }
}
