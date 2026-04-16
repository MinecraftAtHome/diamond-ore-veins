# nvcc -g -maxrregcount 128 --resource-usage -lineinfo -Xptxas -lineinfo -v -O3 -arch=all \
#         main.cu -m64 -o cuda \
#         -DBOINC -Iboinc/ -Lboinc/lib/lin -lcuda -lboinc_api -lboinc -Xptxas -v
#         ls -la

# g++ -o cl opencl_main.cpp -lOpenCL \
#         -O3 \
#         -march=sandybridge \
#         -DCL_HPP_TARGET_OPENCL_VERSION=200 \
#         -Iboinc/ -Lboinc/lib/lin -lcuda -lboinc_api -lboinc_opencl -lboinc

# x86_64-w64-mingw32-g++ -o windows_cl opencl_main.cpp -D_WIN64 -DWIN64 -D_WIN32 -DWIN32 -DCL_HPP_ENABLE_EXCEPTIONS -static-libgcc -static-libstdc++ -L. -lOpenCL -DCL_HPP_TARGET_OPENCL_VERSION=200 -I/usr/include/
# x86_64-w64-mingw32-g++ -o windows_cl opencl_main.cpp \
#   -D_WIN64 -DWIN64 -D_WIN32 -DWIN32 -DBOINC \
#   -static-libgcc -static-libstdc++ \
#   -Iboinc/ -Lboinc/lib/win -lboinc_api -lboinc_opencl -lboinc \
#   -L. -lOpenCL \

x86_64-w64-mingw32-g++ -o windows_cl.exe opencl_main.cpp \
  -D_WIN64 -DWIN64 -D_WIN32 -DWIN32 -DBOINC \
  -Iboinc \
  -Lboinc/lib/win \
  -static-libgcc -static-libstdc++ \
  -Wl,--stack,0x800000 \
  -Wl,--start-group -lboinc_api -lboinc_opencl -lboinc -Wl,--end-group \
  -Lcl -lOpenCL