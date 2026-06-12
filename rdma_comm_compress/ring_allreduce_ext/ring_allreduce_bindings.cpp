#include "ring_allreduce_backend.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "bf16_add_",
      &ring_allreduce::bf16_add_,
      py::arg("dst"),
      py::arg("src"),
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "mee_compressed_bytes",
      &ring_allreduce::mee_compressed_bytes,
      py::arg("n"),
      py::arg("nbits"));
  m.def(
      "mee_compress_bf16",
      &ring_allreduce::mee_compress_bf16,
      py::arg("src"),
      py::arg("dst"),
      py::arg("rand_pool"),
      py::arg("nbits"),
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "mee_decompress_bf16",
      &ring_allreduce::mee_decompress_bf16,
      py::arg("src"),
      py::arg("dst"),
      py::arg("nbits"),
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "mee_decompress_add_bf16",
      &ring_allreduce::mee_decompress_add_bf16,
      py::arg("src"),
      py::arg("dst"),
      py::arg("nbits"),
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "mee_dec_comp_bf16",
      &ring_allreduce::mee_dec_comp_bf16,
      py::arg("recv"),
      py::arg("inp"),
      py::arg("send"),
      py::arg("rand_pool"),
      py::arg("nbits"),
      py::call_guard<py::gil_scoped_release>());

  py::class_<ring_allreduce::RdmaSender>(m, "RdmaSender")
      .def(
          py::init<std::string, std::string, int, int, int, std::size_t, int>(),
          py::arg("server_addr"),
          py::arg("server_addr2"),
          py::arg("port"),
          py::arg("rails"),
          py::arg("gpu"),
          py::arg("total_bytes"),
          py::arg("gid_index") = -1,
          py::call_guard<py::gil_scoped_release>())
      .def("send", &ring_allreduce::RdmaSender::send, py::call_guard<py::gil_scoped_release>());

  py::class_<ring_allreduce::RdmaReceiver>(m, "RdmaReceiver")
      .def(
          py::init<std::string, std::string, int, int, int, std::size_t, int>(),
          py::arg("server_addr"),
          py::arg("server_addr2"),
          py::arg("port"),
          py::arg("rails"),
          py::arg("gpu"),
          py::arg("total_bytes"),
          py::arg("gid_index") = -1,
          py::call_guard<py::gil_scoped_release>())
      .def("recv", &ring_allreduce::RdmaReceiver::recv, py::call_guard<py::gil_scoped_release>());

  py::class_<ring_allreduce::PipelineRdmaSender>(m, "PipelineRdmaSender")
      .def(
          py::init<std::string, std::string, int, int, int, std::size_t, int>(),
          py::arg("server_addr"),
          py::arg("server_addr2"),
          py::arg("port"),
          py::arg("rails"),
          py::arg("gpu"),
          py::arg("total_bytes"),
          py::arg("gid_index") = -1,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "send",
          &ring_allreduce::PipelineRdmaSender::send,
          py::arg("src"),
          py::arg("nbits"),
          py::arg("chunk_size"),
          py::arg("strategy"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "send_compress_bf16",
          &ring_allreduce::PipelineRdmaSender::send_compress_bf16,
          py::arg("src"),
          py::arg("dst"),
          py::arg("rand_pool"),
          py::arg("nbits"),
          py::arg("chunk_size"),
          py::arg("strategy"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "send_dec_comp_bf16",
          &ring_allreduce::PipelineRdmaSender::send_dec_comp_bf16,
          py::arg("recv"),
          py::arg("inp"),
          py::arg("send"),
          py::arg("rand_pool"),
          py::arg("nbits"),
          py::arg("chunk_size"),
          py::arg("strategy"),
          py::call_guard<py::gil_scoped_release>());

  py::class_<ring_allreduce::PipelineRdmaReceiver>(m, "PipelineRdmaReceiver")
      .def(
          py::init<std::string, std::string, int, int, int, std::size_t, int>(),
          py::arg("server_addr"),
          py::arg("server_addr2"),
          py::arg("port"),
          py::arg("rails"),
          py::arg("gpu"),
          py::arg("total_bytes"),
          py::arg("gid_index") = -1,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recv",
          &ring_allreduce::PipelineRdmaReceiver::recv,
          py::arg("dst"),
          py::arg("nbits"),
          py::arg("chunk_size"),
          py::arg("strategy"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recv_decompress_bf16",
          &ring_allreduce::PipelineRdmaReceiver::recv_decompress_bf16,
          py::arg("src"),
          py::arg("dst"),
          py::arg("nbits"),
          py::arg("chunk_size"),
          py::arg("strategy"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recv_dec_comp_bf16",
          &ring_allreduce::PipelineRdmaReceiver::recv_dec_comp_bf16,
          py::arg("recv"),
          py::arg("inp"),
          py::arg("send"),
          py::arg("rand_pool"),
          py::arg("nbits"),
          py::arg("chunk_size"),
          py::arg("strategy"),
          py::call_guard<py::gil_scoped_release>());
}
