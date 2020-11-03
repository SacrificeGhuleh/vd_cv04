CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(ParticleFlowVisualization)

# add libs you need
SET(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)
SET(SRC main.cpp colors.h)

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC} cli.h)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
