TEMPLATE = lib
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += staticlib

QMAKE_CFLAGS_RELEASE -= -O2
QMAKE_CFLAGS_RELEASE += -O3 -funroll-loops -mavx

SOURCES += \
    matrix.c \
    mem.c \
    randomgen.c \
    matrix_nn.c \
    cache.c \
    dataset.c \
    mlp.c \
    mlp_highlevel.c
    
include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    errors.h \
    matrix.h \
    mem.h \
    randomgen.h \
    types.h \
    matrix_nn.h \
    cache.h \
    dataset.h \
    mlp.h \
    mlp_highlevel.h

