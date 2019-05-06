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

target_compile_features(
	${PROJECT_NAME} 
	PUBLIC cxx_std_17
)

target_include_directories(
    ${PROJECT_NAME} 
    PUBLIC 
        deps/CImg-2.5.6
)

target_link_libraries(
    ${PROJECT_NAME} 
)

if(WIN32)
	file(GLOB OPENCV_LIBS deps/opencv/win/lib/*.lib)
	file(GLOB OPENCV_STATIC_LIBS deps/opencv/win/staticlib/*.lib)
	target_include_directories(
		${PROJECT_NAME} 
		PUBLIC 
			deps/opencv/win/include
	)
	target_link_libraries(
		${PROJECT_NAME} 
		dwmapi 
		${OPENCV_LIBS}
		${OPENCV_STATIC_LIBS}
	)
endif(WIN32)

if(APPLE)
	file(GLOB OPENCV_LIBS deps/opencv/mac/lib/*.dylib)
	find_package(X11 REQUIRED)
	find_package(Boost REQUIRED)
	# macOS still does not ship with C++17 filesystem header
	# we need to use boost::filesystem instead
	target_compile_definitions(
		${PROJECT_NAME}
		PUBLIC
			APPLE=1
	)
	target_link_directories(
		${PROJECT_NAME}
		PUBLIC
			${X11_LIBRARY_DIRS}
			${Boost_LIBRARY_DIRS}
	)
	target_include_directories(
		${PROJECT_NAME} 
		PUBLIC 
			deps/opencv/mac/include
			${X11_INCLUDE_DIR}
			${Boost_INCLUDE_DIR}
	)
	target_link_libraries(
		${PROJECT_NAME} 
		screen_capture_lite 
		${X11_LIBRARIES}
		boost_filesystem
		${OPENCV_LIBS}
		"-framework CoreFoundation"
		"-framework CoreGraphics"
		"-framework ImageIO"
		"-framework CoreMedia"
		"-framework CoreVideo"
		"-framework AVFoundation"
		"-framework AppKit"
	)
endif(APPLE)
