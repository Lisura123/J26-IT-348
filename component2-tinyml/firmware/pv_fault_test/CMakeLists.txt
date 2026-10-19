# On-device test and benchmark cores (golden vectors from tests/host_c).
set(PV_FAULT_TEST_GOLDEN "${CMAKE_CURRENT_LIST_DIR}/../../tests/host_c")
if(ESP_PLATFORM)
    idf_component_register(SRCS pv_selftest.c pv_bench.c
                           INCLUDE_DIRS "."
                           PRIV_INCLUDE_DIRS ${PV_FAULT_TEST_GOLDEN}
                           REQUIRES pv_fault)
    target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
endif()
