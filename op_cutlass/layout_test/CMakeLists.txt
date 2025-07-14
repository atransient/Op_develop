
# add_executable(00_layout_mapping_logic 
# 00_layout_mapping_logic.cpp
# )

# target_link_directories(
#     00_layout_mapping_logic PUBLIC
#     ${cu_path}
# )

# target_link_libraries(
#     00_layout_mapping_logic PUBLIC
#     ${CUDA_LIBRARIES}
# )


function(layout_compile file_path)
    get_filename_component(filename_only "${file_path}" NAME)
    string(REPLACE ".cpp" "" tar_name ${filename_only})
    add_executable(${tar_name}  ${filename_only})
    target_include_directories(
                            ${tar_name} PUBLIC
                            ${cutlass_path}
                            ${CUDA_INCLUDE_DIRS}
                        )
    target_link_directories(
                            ${tar_name} PUBLIC
                            ${cu_path}
                        )
    target_link_libraries(
                            ${tar_name} PUBLIC
                            ${CUDA_LIBRARIES}
                        )
endfunction()


file(GLOB CPP_SOURCE_FILES "*.cpp")
message(STATUS "CPP_SOURCE_FILES: ${CPP_SOURCE_FILES}")

foreach(file_path IN LISTS CPP_SOURCE_FILES)
    layout_compile(${file_path})
endforeach()
