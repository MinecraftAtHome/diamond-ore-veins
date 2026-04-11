# nvcc -g -maxrregcount 128 --resource-usage -lineinfo -Xptxas -lineinfo -v -O3 -arch=all \
#         main.cu -m64 -o cuda \
#         -DBOINC -Iboinc/ -Lboinc/lib/lin -lcuda -lboinc_api -lboinc -Xptxas -v
#         ls -la

g++ -o cl opencl_main.cpp -lOpenCL \
        -O3 \
        -march=native \
        -DCL_HPP_TARGET_OPENCL_VERSION=200 \
        -Iboinc/ -Lboinc/lib/lin -lcuda -lboinc_api -lboinc_opencl -lboinc