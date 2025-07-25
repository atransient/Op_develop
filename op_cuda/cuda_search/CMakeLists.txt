function(cuda_compile file_path)
    get_filename_component(filename_only "${file_path}" NAME)
    GET_FILENAME_COMPONENT(tar_name ${filename_only} NAME_WE)
    message(STATUS "tar_name: ${tar_name}")
    add_executable(${tar_name}  ${filename_only})
    target_include_directories(
                            ${tar_name} PUBLIC
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
    target_compile_options(${tar_name} PRIVATE -DNDEBUG -arch sm_90)
endfunction()


file(GLOB CPP_SOURCE_FILES "*.cpp" "*.cu")
# message(STATUS "CPP_SOURCE_FILES: ${CPP_SOURCE_FILES}")

foreach(file_path IN LISTS CPP_SOURCE_FILES)
    cuda_compile(${file_path})
endforeach()