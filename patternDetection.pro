#-------------------------------------------------
#
# Project created by QtCreator 2015-05-05T06:48:17
#
#-------------------------------------------------

QT       += core
QT       += gui
QT       +=  multimedia
TARGET = patternDetection
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

#FOR PC
#QMAKE_CXXFLAGS += -std=c++11


CONFIG += c++11

win32{
    LVT_DEV_DIR  =  "$$system(echo C:\\users\\%username%\\IMTSDev)"
    LVT_MPOS_DIR = "$$system(echo C:\\users\\%username%\\AppData\\Local\\mPOS)"

    #INCLUDEPATH += C:\opencv\build\include
    #LIBS += -L"C:\opencv\build\x86\vc12\bin"
    #LIBS += -lopencv_world300d

    INCLUDEPATH += $${LVT_DEV_DIR}\\opencv\\build\\include
    #INCLUDEPATH += $${LVT_DEV_DIR}\\src\\IMTSProjectDevNewApp\\IMTSLibraries\\IMTSResourceManagerLibrary

    #LIBS += -lopencv_core2411d -lopencv_highgui2411d -lopencv_imgproc2411d -lopencv_objdetect2411d -lopencv_features2d2411d -lopencv_flann2411d -lopencv_calib3d2411d -lopencv_nonfree2411d -lopencv_photo2411d -lopencv_ocl2411d
    LIBS += -L$${LVT_DEV_DIR}\\opencv\\build\\x86\\vc12\\lib -lopencv_world300d
    #LIBS += -L$${LVT_MPOS_DIR}\\debug -lIMTS_WIN_ResourceManagerLibrary
}

SOURCES += main.cpp \
    patternDetection.cpp \
    MatToQImage.cpp

HEADERS += \
    patternDetection.h \
    definitions.h \
    MatToQImage.h

DISTFILES += \
    notes.txt
