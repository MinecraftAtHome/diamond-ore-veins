#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <stdatomic.h>
#import <QuartzCore/QuartzCore.h>

const NSInteger RESULTS_SIZE = 5012;

NS_ASSUME_NONNULL_BEGIN

@interface MetalAdder : NSObject
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) prepareData;
- (void) sendComputeCommand;
@end

NS_ASSUME_NONNULL_END

struct CrunchResource {
	uint64_t seed;
	uint64_t out[RESULTS_SIZE];
	uint32_t result_count;
};
int powersOfTwo[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
int main(int argc, const char * argv[]) {
    uint64_t block_min = 0;
    uint64_t block_max = 1000;
    for (int i = 1; i < argc; i += 2) {
		const char *param = argv[i];
		if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
			sscanf(argv[i + 1], "%llu", &block_min);
		} else if (strcmp(param, "-e") == 0 || strcmp(param, "--end") == 0) {
			sscanf(argv[i + 1], "%llu", &block_max);
		} 
        else {
			fprintf(stderr,"Unknown parameter: %s\n", param);
        }
    }
    FILE* seedsout = fopen("seeds.txt", "a");
    @autoreleasepool {
        NSError* error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	NSURL *libraryURL = [[NSBundle mainBundle] URLForResource:@"crunch"
                                                withExtension:@"metallib"];
	NSError *libraryError = nil;
    	id <MTLLibrary> library = [device newLibraryWithURL:libraryURL
                                                  error:&libraryError];
	id<MTLFunction> filterFunction = [library newFunctionWithName:@"crunch"];

	id<MTLComputePipelineState> filterFunctionPSO = [device newComputePipelineStateWithFunction: filterFunction error:&error];

	NSInteger threadgroupMaxThreads = filterFunctionPSO.maxTotalThreadsPerThreadgroup;
	printf("threadgroupMaxThreads: %lu\n", threadgroupMaxThreads);
	// if((threadgroupMaxThreads & (threadgroupMaxThreads - 1)) != 0) {
	// 	printf("Finding new max threads...\n");
	// 	for(int i = sizeof(powersOfTwo)/sizeof(int) - 1; i >= 0; i--){
	// 		if(powersOfTwo[i] < threadgroupMaxThreads){
	// 			threadgroupMaxThreads = powersOfTwo[i];
	// 			break;
	// 		}
	// 	}
	// }
	// printf("threadgroupMaxThreads: %lu\n", threadgroupMaxThreads);
	MTLSize threadgroupSize = MTLSizeMake(threadgroupMaxThreads, 1, 1);

	id<MTLCommandQueue> commandQueue = [device newCommandQueue];
	
	id<MTLBuffer> crunch_res = [device newBufferWithLength:sizeof(struct CrunchResource) options:MTLResourceStorageModeShared];
	struct CrunchResource* crunch_res_ptr = (struct CrunchResource*)crunch_res.contents;

	MTLSize iterationSize = MTLSizeMake(1ULL << 16, 1ULL << 16, 1);
	printf("begin execution\n");
	uint64_t chkpoint = 0;
	CFTimeInterval startTime = CACurrentMediaTime();
	for(uint64_t s = block_min; s < block_max; s++) {
		printf("next launch!\n");
		crunch_res_ptr->seed = s * (1ULL << 32);
		id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		[computeEncoder setComputePipelineState:filterFunctionPSO];
		[computeEncoder setBuffer:crunch_res offset:0 atIndex:0];
		[computeEncoder dispatchThreads:iterationSize threadsPerThreadgroup:threadgroupSize];
		[computeEncoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		uint32_t result_count = crunch_res_ptr->result_count;
		if(result_count > 0){
			for(unsigned long long index = 0; index < result_count; index++){
				printf("HIT: %llu\n", crunch_res_ptr->out[index]);
				fprintf(seedsout, "%llu\n", crunch_res_ptr->out[index]);
				crunch_res_ptr->out[index] = (uint64_t)0;
			}
			fflush(seedsout);

		}
		crunch_res_ptr->result_count = (uint32_t)0;
		chkpoint++;
		if(chkpoint == 5) {
			CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
			uint64_t seeds_calced = (s - block_min) * (1ULL << 32);
			uint64_t seeds_remain = (block_max - s) * (1ULL << 32);
			double sps = seeds_calced/elapsedTime;
			printf("sps: %f, eta: %fs", sps, (double)seeds_remain / sps);

		}

	}
	NSLog(@"Execution finished");
    }
    return 0;
}
