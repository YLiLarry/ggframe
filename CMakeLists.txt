cmake_minimum_required(VERSION 3.8)
project(ggframe VERSION 1.0)

option(build_example "build examples?" ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

file(GLOB SOURCES src/*.cc)
file(GLOB HEADERS include/*.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES})


install (TARGETS ${PROJECT_NAME}  
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
)
install(FILES ${HEADERS} DESTINATION include/${PROJECT_NAME})

target_link_directories(
    ${PROJECT_NAME}
    PUBLIC
		deps/opencv/lib
)
target_include_directories(
    ${PROJECT_NAME} 
    PUBLIC 
        deps/CImg-2.5.6
        deps/opencv/win/include
)

if(WIN32)
file(GLOB OPENCV_LIBS deps/opencv/win/lib/*.lib)
file(GLOB OPENCV_STATIC_LIBS deps/opencv/win/staticlib/*.lib)
endif(WIN32)

target_link_libraries(
    ${PROJECT_NAME} 
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

if(WIN32)
	target_link_libraries(
		${PROJECT_NAME} 
		dwmapi 
		${OPENCV_LIBS}
		${OPENCV_STATIC_LIBS}
	)
endif(WIN32)

if(APPLE)
endif(APPLE)
