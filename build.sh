x86_64-w64-mingw32-g++ -o windows_cl.exe opencl_main.cpp \
  -D_WIN64 -DWIN64 -D_WIN32 -DWIN32 \
  -Iboinc \
  -Lboinc/lib/win \
  -static-libgcc -static-libstdc++ \
  -lboinc_api -lboinc_opencl -lboinc \
  -Lcl -lOpenCL