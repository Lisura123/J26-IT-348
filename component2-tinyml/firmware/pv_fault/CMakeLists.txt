# PV fault classifier (Component 2). Two uses:
#  - ESP-IDF component (ESP_PLATFORM set by the IDF build system), and
#  - plain host library for local checks: cmake -S firmware/pv_fault -B build
# Sources are portable C99; generated headers are not edited by hand.
set(PV_FAULT_SRCS
    pv_fault.c pv_mlp_f32.c pv_mlp_int8.c
    pv_api.c pv_sensor.c pv_confirm.c pv_relay.c pv_message.c pv_pipeline.c)
set(PV_FAULT_INCLUDES "." "generated" "selfcheck")

if(ESP_PLATFORM)
    idf_component_register(SRCS ${PV_FAULT_SRCS} INCLUDE_DIRS ${PV_FAULT_INCLUDES})
    # No fused multiply-add: keeps Float32 results equal to the reference (QUANTIZATION.md 5).
    target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
else()
    cmake_minimum_required(VERSION 3.16)
    project(pv_fault C)
    add_library(pv_fault STATIC ${PV_FAULT_SRCS})
    target_include_directories(pv_fault PUBLIC ${PV_FAULT_INCLUDES})
    set_target_properties(pv_fault PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_compile_options(pv_fault PRIVATE -Wall -Wextra -Werror -pedantic -ffp-contract=off)
    target_link_libraries(pv_fault PUBLIC m)
endif()
