# ESP-IDF application (Component 2). Build: idf.py set-target <chip> && idf.py build
# The board is not known yet; no target or board is assumed here.
cmake_minimum_required(VERSION 3.16)
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../pv_fault" "${CMAKE_CURRENT_LIST_DIR}/../../pv_fault_test")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(pv_selftest_app)
