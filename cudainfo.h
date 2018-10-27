#ifndef CZ_CUDAINFO_H
#define CZ_CUDAINFO_H

#ifdef __cplusplus
extern "C" {
#endif

/*!	\brief Device compute mode.
 * */
enum CZComputeMode {
	CZComputeModeUnknown = -1,		/*!< Unknown compute mode. */
	CZComputeModeDefault = 0,		/*!< Default compute mode. */
	CZComputeModeExclusive,			/*!< Compute-exclusive mode. */
	CZComputeModeProhibited,		/*!< Compute-prohibited mode. */
};

/*!	\brief Information about CUDA-device core.
 * */
struct CZDeviceInfoCore {
	int		regsPerBlock;		/*!< Total number of registers available per block. */
	int		SIMDWidth;		/*!< Warp size. */
	int		maxThreadsPerBlock;	/*!< Maximum number of threads per block. */
	int		maxThreadsDim[3];	/*!< Maximum sizes of each dimension of a block. */
	int		maxGridSize[3];		/*!< Maximum sizes of each dimension of a grid. */
	int		clockRate;		/*!< Clock frequency in kilohertz. */
	int		muliProcCount;		/*!< Number of mutiprocessors in GPU. */
	int		watchdogEnabled;	/*!< Has run time limit for kernels executed. */
	int		integratedGpu;		/*!< 1 if the device is an integrated GPU and 0 if it is a discrete component. */
	int		concurrentKernels;	/*!< 1 if the device supports executing multiple kernels within the same context simultaneously. */
	int		computeMode;		/*!< Current compute mode. See enum #CZComputeMode. */
	int		pciBusID;		/*!< PCI bus identifier of the device. */
	int		pciDeviceID;		/*!< PCI device (sometimes called slot) identifier of the device. */
	int		pciDomainID;		/*!< PCI domain identifier of the device. */
	int		maxThreadsPerMultiProcessor;	/*!< Number of maximum resident threads per multiprocessor. */
	int		cudaCores;		/*!< Number of CUDA cores. */
	int		streamPrioritiesSupported;	/*!< Stream priorities supported. */
};

/*!	\brief Information about CUDA-device memory.
 * */
struct CZDeviceInfoMem {
	size_t		totalGlobal;		/*!< Total amount of global memory available on the device in bytes. */
	size_t		sharedPerBlock;		/*!< Total amount of shared memory available per block in bytes. */
	size_t		maxPitch;		/*!< Maximum pitch allowed by the memory copy functions that involve memory region allocated through cudaMallocPitch()/cuMemAllocPitch() */
	size_t		totalConst;		/*!< Total amount of constant memory available on the device in bytes. */
	size_t		textureAlignment;	/*!< Texture base addresses that are aligned to textureAlignment bytes do not need an offset applied to texture fetches. */
	size_t		texture1D[1];		/*!< 1D texture size. */
	size_t		texture2D[2];		/*!< 2D texture size. */
	size_t		texture3D[3];		/*!< 3D texture size. */
	int		gpuOverlap;		/*!< 1 if the device can concurrently copy memory between host and device while executing a kernel, or 0 if not. */
	int		mapHostMemory;		/*!< 1 if device can map host memory. */
	int		errorCorrection;	/*!< 1 if error correction is enabled on the device. */
	int		asyncEngineCount;	/*!< 1 if unidirectional, 2 if bitirectional, 0 if not supported. */
	int		unifiedAddressing;	/*!< 1 if the device shares a unified address space with the host and 0 otherwise. */
	int		memoryClockRate;	/*!< Peak memory clock frequency in kilohertz. */
	int		memoryBusWidth;		/*!< Memory bus width in bits. */
	int		l2CacheSize;		/*!< L2 cache size in bytes. */
};

/*!	\brief Information about CUDA-device bandwidth.
 * */
struct CZDeviceInfoBand {
	float		copyHDPage;		/*!< Copy rate from host pageable to device memory in KB/s. */
	float		copyHDPin;		/*!< Copy rate from host pinned to device memory in KB/s. */
	float		copyDHPage;		/*!< Copy rate from device to host pageable memory in KB/s. */
	float		copyDHPin;		/*!< Copy rate from device to host pinned memory in KB/s. */
	float		copyDD;			/*!< Copy rate from device to device memory in KB/s. */
	/* Service part of structure. */
	void		*localData;
};

/*!	\brief Information about CUDA-device performance.
 * */
struct CZDeviceInfoPerf {
	float		calcFloat;		/*!< Single-precision float point calculations performance in KFOPS. */
	float		calcDouble;		/*!< Double-precision float point calculations performance in KFOPS. */
	float		calcInteger32;		/*!< 32-bit integer calculations performance in KOPS. */
	float		calcInteger24;		/*!< 24-bit integer calculations performance in KOPS. */
	float		calcInteger64;		/*!< 64-bit integer calculations performance in KOPS. */
};

/*!	\brief Information about CUDA-device.
 * */
struct CZDeviceInfo {
	int		num;			/*!< Device index. */
	int		heavyMode;		/*!< Heavy test mode flag. */
	char		deviceName[256];	/*!< ASCII string identifying the device. */
	int		major;			/*!< Major revision numbers defining the device's compute capability. */
	int		minor;			/*!< Minor revision numbers defining the device's compute capability. */
	char		*drvVersion;		/*!< Driver version string. */
	int		drvDllVer;		/*!< Driver Dll version. */
	char		*drvDllVerStr;		/*!< Driver Dll version string. */
	int		rtDllVer;		/*!< Runtime Dll version. */
	char		*rtDllVerStr;		/*!< Runtime Dll version string. */
	int		tccDriver;		/*!< 1 if the device is using a TCC driver or 0 if not. */
	struct CZDeviceInfoCore	core;
	struct CZDeviceInfoMem	mem;
	struct CZDeviceInfoBand	band;
	struct CZDeviceInfoPerf	perf;
};

bool CZCudaCheck(void);
int CZCudaDeviceFound(void);
int CZCudaReadDeviceInfo(struct CZDeviceInfo *info, int num);
int CZCudaCalcDeviceSelect(struct CZDeviceInfo *info);
int CZCudaPrepareDevice(struct CZDeviceInfo *info);
int CZCudaCalcDeviceBandwidth(struct CZDeviceInfo *info);
int CZCudaCalcDevicePerformance(struct CZDeviceInfo *info);
int CZCudaCleanDevice(struct CZDeviceInfo *info);

#ifdef __cplusplus
}
#endif

#endif//CZ_CUDAINFO_H

