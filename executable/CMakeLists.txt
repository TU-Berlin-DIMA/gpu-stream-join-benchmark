add_executable(debug_datagen debug_datagen.cpp)
target_link_libraries(debug_datagen PRIVATE sjb)

add_executable(runExperiment runExperiment.cpp)
target_link_libraries(runExperiment PRIVATE sjb experiment)

add_executable(standalone standalone.cpp)
target_link_libraries(standalone PRIVATE sjb experiment)