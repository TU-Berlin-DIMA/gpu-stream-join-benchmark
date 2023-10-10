# Set Sources and Headers for SJB
add_subdirectory(sjb)
get_source(sjb SJB_SOURCE_FILES)

# Add SJB Library
add_library(sjb SHARED ${SJB_SOURCE_FILES})
target_include_directories(sjb PUBLIC ${CMAKE_SOURCE_DIR}/include/)

# Link to libraries to SJB
target_link_libraries(sjb PUBLIC ${LIBRARIES})
set_property(TARGET sjb PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# ----
# Set Sources and Headers
add_subdirectory(experiment)
get_source(experiment EXPERIMENT_SOURCE_FILES)

# Add Library
add_library(experiment SHARED ${EXPERIMENT_SOURCE_FILES})
target_include_directories(experiment PUBLIC ${CMAKE_SOURCE_DIR}/include/)

# Link to libraries
target_link_libraries(experiment sjb)