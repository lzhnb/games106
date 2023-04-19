/*
* Class wrapping access to the swap chain
* 
* A swap chain is a collection of framebuffers used for rendering and presentation to the windowing system
*
* Copyright (C) 2016-2021 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanSwapChain.h"

/** @brief Creates the platform specific surface abstraction of the native platform window used for presentation */	
#if defined(VK_USE_PLATFORM_WIN32_KHR)
void VulkanSwapChain::initSurface(void* platformHandle, void* platformWindow)
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
void VulkanSwapChain::initSurface(ANativeWindow* window)
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
void VulkanSwapChain::initSurface(IDirectFB* dfb, IDirectFBSurface* window)
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
void VulkanSwapChain::initSurface(wl_display *display, wl_surface *window)
#elif defined(VK_USE_PLATFORM_XCB_KHR)
void VulkanSwapChain::initSurface(xcb_connection_t* connection, xcb_window_t window)
#elif (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
void VulkanSwapChain::initSurface(void* view)
#elif (defined(_DIRECT2DISPLAY) || defined(VK_USE_PLATFORM_HEADLESS_EXT))
void VulkanSwapChain::initSurface(uint32_t width, uint32_t height)
#endif
{
	VkResult err = VK_SUCCESS;

	// Create the os-specific surface
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	surfaceCreateInfo.hinstance = (HINSTANCE)platformHandle;
	surfaceCreateInfo.hwnd = (HWND)platformWindow;
	err = vkCreateWin32SurfaceKHR(instance, &surfaceCreateInfo, nullptr, &surface);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
	VkAndroidSurfaceCreateInfoKHR surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
	surfaceCreateInfo.window = window;
	err = vkCreateAndroidSurfaceKHR(instance, &surfaceCreateInfo, NULL, &surface);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
	VkIOSSurfaceCreateInfoMVK surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK;
	surfaceCreateInfo.pNext = NULL;
	surfaceCreateInfo.flags = 0;
	surfaceCreateInfo.pView = view;
	err = vkCreateIOSSurfaceMVK(instance, &surfaceCreateInfo, nullptr, &surface);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
	VkMacOSSurfaceCreateInfoMVK surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
	surfaceCreateInfo.pNext = NULL;
	surfaceCreateInfo.flags = 0;
	surfaceCreateInfo.pView = view;
	err = vkCreateMacOSSurfaceMVK(instance, &surfaceCreateInfo, NULL, &surface);
#elif defined(_DIRECT2DISPLAY)
	createDirect2DisplaySurface(width, height);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
	VkDirectFBSurfaceCreateInfoEXT surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_DIRECTFB_SURFACE_CREATE_INFO_EXT;
	surfaceCreateInfo.dfb = dfb;
	surfaceCreateInfo.surface = window;
	err = vkCreateDirectFBSurfaceEXT(instance, &surfaceCreateInfo, nullptr, &surface);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	VkWaylandSurfaceCreateInfoKHR surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
	surfaceCreateInfo.display = display;
	surfaceCreateInfo.surface = window;
	err = vkCreateWaylandSurfaceKHR(instance, &surfaceCreateInfo, nullptr, &surface);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	VkXcbSurfaceCreateInfoKHR surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
	surfaceCreateInfo.connection = connection;
	surfaceCreateInfo.window = window;
	err = vkCreateXcbSurfaceKHR(instance, &surfaceCreateInfo, nullptr, &surface);
#elif defined(VK_USE_PLATFORM_HEADLESS_EXT)
	VkHeadlessSurfaceCreateInfoEXT surfaceCreateInfo = {};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
	PFN_vkCreateHeadlessSurfaceEXT fpCreateHeadlessSurfaceEXT = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
	if (!fpCreateHeadlessSurfaceEXT){
		vks::tools::exitFatal("Could not fetch function pointer for the headless extension!", -1);
	}
	err = fpCreateHeadlessSurfaceEXT(instance, &surfaceCreateInfo, nullptr, &surface);
#endif

	if (err != VK_SUCCESS) {
		vks::tools::exitFatal("Could not create surface!", err);
	}

	// Get available queue family properties
	uint32_t queueCount;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, NULL);
	assert(queueCount >= 1);

	std::vector<VkQueueFamilyProperties> queueProps(queueCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queueProps.data());

	// Iterate over each queue to learn whether it supports presenting:
	// Find a queue with present support
	// Will be used to present the swap chain images to the windowing system
	// 通过检查队列是否具有 VK_QUEUE_GRAPHICS_BIT 的同级循环调用 vkGetPhysicalDeviceSurfaceSupportKHR
	// 函数来查找带有呈现图像到窗口表面能力的队列族。
	std::vector<VkBool32> supportsPresent(queueCount);
	for (uint32_t i = 0; i < queueCount; i++) 
	{
		fpGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &supportsPresent[i]);
	}

	// Search for a graphics and a present queue in the array of queue
	// families, try to find one that supports both
	uint32_t graphicsQueueNodeIndex = UINT32_MAX;
	uint32_t presentQueueNodeIndex = UINT32_MAX;
	for (uint32_t i = 0; i < queueCount; i++) 
	{
		if ((queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) 
		{
			if (graphicsQueueNodeIndex == UINT32_MAX) 
			{
				graphicsQueueNodeIndex = i;
			}

			if (supportsPresent[i] == VK_TRUE) 
			{
				graphicsQueueNodeIndex = i;
				presentQueueNodeIndex = i;
				break;
			}
		}
	}
	if (presentQueueNodeIndex == UINT32_MAX) 
	{	
		// If there's no queue that supports both present and graphics
		// try to find a separate present queue
		for (uint32_t i = 0; i < queueCount; ++i) 
		{
			if (supportsPresent[i] == VK_TRUE) 
			{
				presentQueueNodeIndex = i;
				break;
			}
		}
	}

	// Exit if either a graphics or a presenting queue hasn't been found
	if (graphicsQueueNodeIndex == UINT32_MAX || presentQueueNodeIndex == UINT32_MAX) 
	{
		vks::tools::exitFatal("Could not find a graphics and/or presenting queue!", -1);
	}

	// todo : Add support for separate graphics and presenting queue
	if (graphicsQueueNodeIndex != presentQueueNodeIndex) 
	{
		vks::tools::exitFatal("Separate graphics and presenting queues are not supported yet!", -1);
	}

	queueNodeIndex = graphicsQueueNodeIndex;

	// Get list of supported surface formats
	uint32_t formatCount;
	VK_CHECK_RESULT(fpGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, NULL));
	assert(formatCount > 0);

	std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount); // 表面格式（像素格式，颜色空间）
	VK_CHECK_RESULT(fpGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, surfaceFormats.data()));

	// If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
	// there is no preferred format, so we assume VK_FORMAT_B8G8R8A8_UNORM
	// 选择合适的表面格式（颜色深度）
	if ((formatCount == 1) && (surfaceFormats[0].format == VK_FORMAT_UNDEFINED))
	{
		colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
		colorSpace = surfaceFormats[0].colorSpace;
	}
	else
	{
		// iterate over the list of available surface format and
		// check for the presence of VK_FORMAT_B8G8R8A8_UNORM
		bool found_B8G8R8A8_UNORM = false;
		for (auto&& surfaceFormat : surfaceFormats)
		{
			if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
			{
				colorFormat = surfaceFormat.format;
				colorSpace = surfaceFormat.colorSpace;
				found_B8G8R8A8_UNORM = true;
				break;
			}
		}

		// in case VK_FORMAT_B8G8R8A8_UNORM is not available
		// select the first available color format
		if (!found_B8G8R8A8_UNORM)
		{
			colorFormat = surfaceFormats[0].format;
			colorSpace = surfaceFormats[0].colorSpace;
		}
	}

}

/**
* Set instance, physical and logical device to use for the swapchain and get all required function pointers
* 
* @param instance Vulkan instance to use
* @param physicalDevice Physical device used to query properties and formats relevant to the swapchain
* @param device Logical representation of the device to create the swapchain for
*
*/
void VulkanSwapChain::connect(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device)
{
	this->instance = instance;
	this->physicalDevice = physicalDevice;
	this->device = device;
	fpGetPhysicalDeviceSurfaceSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfaceSupportKHR"));
	fpGetPhysicalDeviceSurfaceCapabilitiesKHR =  reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
	fpGetPhysicalDeviceSurfaceFormatsKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
	fpGetPhysicalDeviceSurfacePresentModesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"));

	fpCreateSwapchainKHR = reinterpret_cast<PFN_vkCreateSwapchainKHR>(vkGetDeviceProcAddr(device, "vkCreateSwapchainKHR"));
	fpDestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(vkGetDeviceProcAddr(device, "vkDestroySwapchainKHR"));
	fpGetSwapchainImagesKHR = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(vkGetDeviceProcAddr(device, "vkGetSwapchainImagesKHR"));
	fpAcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(vkGetDeviceProcAddr(device, "vkAcquireNextImageKHR"));
	fpQueuePresentKHR = reinterpret_cast<PFN_vkQueuePresentKHR>(vkGetDeviceProcAddr(device, "vkQueuePresentKHR"));
}

/** 
* Create the swapchain and get its images with given width and height
* 
* @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
* @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
* @param vsync (Optional) Can be used to force vsync-ed rendering (by using VK_PRESENT_MODE_FIFO_KHR as presentation mode)
*/
void VulkanSwapChain::create(uint32_t *width, uint32_t *height, bool vsync, bool fullscreen)
{
	// Store the current swap chain handle so we can use it later on to ease up recreation
	VkSwapchainKHR oldSwapchain = swapChain;

	// Get physical device surface properties and formats
	VkSurfaceCapabilitiesKHR surfCaps; // 基础表面属性（交换链的最小戯最大图像数量，最小戯最大图像宽度、高度）
	VK_CHECK_RESULT(fpGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfCaps));

	// Get available present modes
	uint32_t presentModeCount;
	VK_CHECK_RESULT(fpGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, NULL));
	assert(presentModeCount > 0);

	std::vector<VkPresentModeKHR> presentModes(presentModeCount); // 可用的呈现模式
	VK_CHECK_RESULT(fpGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data()));

	// 交换范围（交换链中的图像的分辨率）
	// 几乎总是和我们要显示图像的窗口的分辨率相同。
	VkExtent2D swapchainExtent = {};
	// If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
	if (surfCaps.currentExtent.width == (uint32_t)-1)
	{
		// If the surface size is undefined, the size is set to
		// the size of the images requested.
		swapchainExtent.width = *width;
		swapchainExtent.height = *height;
	}
	else
	{
		// If the surface size is defined, the swap chain size must match
		swapchainExtent = surfCaps.currentExtent;
		*width = surfCaps.currentExtent.width;
		*height = surfCaps.currentExtent.height;
	}


	// Select a present mode for the swapchain

	// The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
	// This mode waits for the vertical blank ("v-sync")
	// 呈现模式（交换链中最重要的设置），决定了什么条件下图像才会显示到屏幕。
	// 交换链变成一个先进先出的队列，每次从队列头部取出一张图像进行显示，应用程序渲染的
	// 图像提交给交换链后，会被放在队列尾部。当队列为满时，应用程序需要进行等待。这一模式
	// 非常类似现在常用的垂直同步。刷新显示的仕科也被叫做垂直回扫。
	// 四种模式中只有这个模式保证一定可用，其他需要查找判断
	VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;

	// 还有一个 VK_PRESENT_MODE_FIFO_RELAXED_KHR 与 VK_PRESENT_MODE_FIFO_KHR 的唯一
	// 区是，如果应用程序延迟，导致交换链的队列在上一次垂直回扫时为空，那么如果应用程序在
	// 下一次垂直回扫前提交图像，图像会立即被显示。这一模式可能会导致撕裂现象。

	// If v-sync is not requested, try to find a mailbox mode
	// It's the lowest latency non-tearing present mode available
	if (!vsync)
	{
		for (size_t i = 0; i < presentModeCount; i++)
		{
			// 是 VK_PRESENT_MODE_FIFO_KHR 的另一个变种。它不会在交换链的队列满时阻塞
			// 应用程序，队列中的图像会被直接替换成应用程序新提交的图像。这一模式可以用来
			// 实现三倍缓冲，避免撕裂现象的同时减小了延迟问题。
			if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
				break;
			}
			// 应用程序提交的图像会被立即传输到屏幕上，可能会导致撕裂现象
			if (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
			{
				swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
			}
		}
	}

	// Determine the number of images
	// 确定交换链的图像数量（通常由硬件 GPU 决定）
	// 使用交换链支持的最小图像个数 +1 数量的图像来实现三倍缓冲
	uint32_t desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1;
#if (defined(VK_USE_PLATFORM_MACOS_MVK) && defined(VK_EXAMPLE_XCODE_GENERATED))
	// SRS - Work around known MoltenVK issue re 2x frame rate when vsync (VK_PRESENT_MODE_FIFO_KHR) enabled
	struct utsname sysInfo;
	uname(&sysInfo);
	// SRS - When vsync is on, use minImageCount when not in fullscreen or when running on Apple Silcon
	// This forces swapchain image acquire frame rate to match display vsync frame rate
	if (vsync && (!fullscreen || strcmp(sysInfo.machine, "arm64") == 0))
	{
		desiredNumberOfSwapchainImages = surfCaps.minImageCount;
	}
#endif
	// 如果 maxImageCount 为 0 则表明，只要内存满足，我们可以使用任意数量的图像。
	if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount))
	{
		desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
	}

	// Find the transformation of the surface
	VkSurfaceTransformFlagsKHR preTransform;
	if (surfCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		// We prefer a non-rotated transform
		preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		preTransform = surfCaps.currentTransform;
	}

	// Find a supported composite alpha format (not all devices support alpha opaque)
	VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	// Simply select the first composite alpha format available
	std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
		VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
	};
	for (auto& compositeAlphaFlag : compositeAlphaFlags) {
		if (surfCaps.supportedCompositeAlpha & compositeAlphaFlag) {
			compositeAlpha = compositeAlphaFlag;
			break;
		};
	}

	VkSwapchainCreateInfoKHR swapchainCI = {};
	swapchainCI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	// 交换链绑定表面后，还需要指定有关交换链图像的信息
	swapchainCI.surface = surface;
	swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
	swapchainCI.imageFormat = colorFormat; // 颜色格式
	swapchainCI.imageColorSpace = colorSpace; // 色彩空间
	swapchainCI.imageExtent = { swapchainExtent.width, swapchainExtent.height };
	swapchainCI.imageArrayLayers = 1; // 用于指定每个图像所包含的层次。通常值为 1（VR 例外）
	swapchainCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	// 指定多个队列族使用交换链图像的方式。这一设置对于图形队列和呈现队列不是同一个队列的情况
	// 有着很大影响。我们通过图形队列在交换链图像上进行绘制操作，然后将图像提交给呈现队列来显示。
	// 有两种控制在多个队列访问图像的方式:
	// VK_SHARING_MODE_EXCLUSIVE: 一张图像同一时间只能被一个队列族所拥有，在另一队列族使用它之前，
	// 							  必须显示地改变图像所有权。这一模式下性能表现最佳。
	// VK_SHARING_MODE_CONCURRENT: 图像可以在多个队列族间使用，不需要显示地改变图像所有权。
	// 如果图形队队列族和呈现队列族是同一个队列族（大部分情况），则不能使用协同模式
	// 系统模式需要指定至少两个不同的队列族
	swapchainCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	// 可以为交换链中的图像指定一个固定的变换操作
	swapchainCI.preTransform = (VkSurfaceTransformFlagBitsKHR)preTransform;
	// 用于指定 alpha 通道是否被用来和窗口系统中的其他窗口进行混合操作
	// 通常设置为 VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR 忽略掉 alpha 通道
	swapchainCI.compositeAlpha = compositeAlpha;
	swapchainCI.presentMode = swapchainPresentMode; // 用于设置呈现模式
	// Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
	swapchainCI.clipped = VK_TRUE;
	// Setting oldSwapChain to the saved handle of the previous swapchain aids in resource reuse and makes sure that we can still present already acquired images
	// 需要指定，因为在程序运行过程中交换链可能会失效
	swapchainCI.oldSwapchain = oldSwapchain;
	swapchainCI.queueFamilyIndexCount = 0;

	// Enable transfer source on swap chain images if supported
	if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
		swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	// Enable transfer destination on swap chain images if supported
	if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
		swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	VK_CHECK_RESULT(fpCreateSwapchainKHR(device, &swapchainCI, nullptr, &swapChain));

	// If an existing swap chain is re-created, destroy the old swap chain
	// This also cleans up all the presentable images
	if (oldSwapchain != VK_NULL_HANDLE) 
	{ 
		for (uint32_t i = 0; i < imageCount; i++)
		{
			vkDestroyImageView(device, buffers[i].view, nullptr);
		}
		fpDestroySwapchainKHR(device, oldSwapchain, nullptr);
	}
	// 获取交换链图像的数量
	VK_CHECK_RESULT(fpGetSwapchainImagesKHR(device, swapChain, &imageCount, NULL));

	// Get the swap chain images
	// images 是 vector<VkImage> 用于存储这些图像句柄，图像句柄用来进行渲染操作
	images.resize(imageCount);
	// 然后分配数组空间，获得交换链图像句柄
	VK_CHECK_RESULT(fpGetSwapchainImagesKHR(device, swapChain, &imageCount, images.data()));

	// Get the swap chain buffers containing the image and imageview
	// 使用任何 VkImage 对象，包括处于交换链中的，处于渲染管线中的，都需要我们创建一个 VkImageView 对象来绑定访问它。
	// 图像视图描述了访问图像的方式，以及图像的哪一部分可以被访问。
	buffers.resize(imageCount);
	for (uint32_t i = 0; i < imageCount; i++)
	{
		VkImageViewCreateInfo colorAttachmentView = {};
		colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		colorAttachmentView.pNext = NULL;
		colorAttachmentView.flags = 0;
		// ViewType 和 format 成员变量用于指定图像数据的解释方式。ViewType 成员变量
		// 用于指定图像被看作是一维纹理、二维纹理、三维纹理还是立方体贴图。
		colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorAttachmentView.format = colorFormat;
		// components 成员变量用于进行图像颜色通道的映射。比如，对于单色纹理，我们可以将所有
		// 颜色通道映射到红色通道。我们也可以直接将颜色通道的值映射为常数 0 或 1。
		colorAttachmentView.components = {
			VK_COMPONENT_SWIZZLE_R,
			VK_COMPONENT_SWIZZLE_G,
			VK_COMPONENT_SWIZZLE_B,
			VK_COMPONENT_SWIZZLE_A
		};
		// subresourceRange 成员变量用于指定图像的用途和图像的哪一部分可以被访问。
		// 在这里，我们的图像被用作渲染目标，并且没有细分级别，只存在一个图层
		colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorAttachmentView.subresourceRange.baseMipLevel = 0;
		colorAttachmentView.subresourceRange.levelCount = 1;
		colorAttachmentView.subresourceRange.baseArrayLayer = 0;
		colorAttachmentView.subresourceRange.layerCount = 1;

		buffers[i].image = images[i];
		colorAttachmentView.image = buffers[i].image;

		// 调用 vkCreateImageView 创建图像视图
		VK_CHECK_RESULT(vkCreateImageView(device, &colorAttachmentView, nullptr, &buffers[i].view));
	}
}

/** 
* Acquires the next image in the swap chain
*
* @param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
* @param imageIndex Pointer to the image index that will be increased if the next image could be acquired
*
* @note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
*
* @return VkResult of the image acquisition
*/
VkResult VulkanSwapChain::acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t *imageIndex)
{
	// By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is thrown
	// With that we don't have to handle VK_NOT_READY
	// 第一个参数是使用的逻辑设备对象，第二个参数是我们要获取图像的交换链，第三个参数是图像获取的超时时间
	// 最后一个参数用于输出可用的交换链图像的索引
	return fpAcquireNextImageKHR(device, swapChain, UINT64_MAX, presentCompleteSemaphore, (VkFence)nullptr, imageIndex);
}

/**
* Queue an image for presentation
*
* @param queue Presentation queue for presenting the image
* @param imageIndex Index of the swapchain image to queue for presentation
* @param waitSemaphore (Optional) Semaphore that is waited on before the image is presented (only used if != VK_NULL_HANDLE)
*
* @return VkResult of the queue presentation
*/
VkResult VulkanSwapChain::queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore)
{
	// 来配置呈现信号
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = NULL;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapChain;
	presentInfo.pImageIndices = &imageIndex;
	// Check if a wait semaphore has been specified to wait for before presenting the image
	if (waitSemaphore != VK_NULL_HANDLE)
	{
		// 用于指定开始呈现操作需要等待的信号量。
		presentInfo.pWaitSemaphores = &waitSemaphore;
		presentInfo.waitSemaphoreCount = 1;
	}
	return fpQueuePresentKHR(queue, &presentInfo);
}


/**
* Destroy and free Vulkan resources used for the swapchain
*/
void VulkanSwapChain::cleanup()
{
	if (swapChain != VK_NULL_HANDLE)
	{
		for (uint32_t i = 0; i < imageCount; i++)
		{
			vkDestroyImageView(device, buffers[i].view, nullptr);
		}
	}
	if (surface != VK_NULL_HANDLE)
	{
		fpDestroySwapchainKHR(device, swapChain, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
	}
	surface = VK_NULL_HANDLE;
	swapChain = VK_NULL_HANDLE;
}

#if defined(_DIRECT2DISPLAY)
/**
* Create direct to display surface
*/	
void VulkanSwapChain::createDirect2DisplaySurface(uint32_t width, uint32_t height)
{
	uint32_t displayPropertyCount;
		
	// Get display property
	vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &displayPropertyCount, NULL);
	VkDisplayPropertiesKHR* pDisplayProperties = new VkDisplayPropertiesKHR[displayPropertyCount];
	vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &displayPropertyCount, pDisplayProperties);

	// Get plane property
	uint32_t planePropertyCount;
	vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &planePropertyCount, NULL);
	VkDisplayPlanePropertiesKHR* pPlaneProperties = new VkDisplayPlanePropertiesKHR[planePropertyCount];
	vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &planePropertyCount, pPlaneProperties);

	VkDisplayKHR display = VK_NULL_HANDLE;
	VkDisplayModeKHR displayMode;
	VkDisplayModePropertiesKHR* pModeProperties;
	bool foundMode = false;

	for(uint32_t i = 0; i < displayPropertyCount;++i)
	{
		display = pDisplayProperties[i].display;
		uint32_t modeCount;
		vkGetDisplayModePropertiesKHR(physicalDevice, display, &modeCount, NULL);
		pModeProperties = new VkDisplayModePropertiesKHR[modeCount];
		vkGetDisplayModePropertiesKHR(physicalDevice, display, &modeCount, pModeProperties);

		for (uint32_t j = 0; j < modeCount; ++j)
		{
			const VkDisplayModePropertiesKHR* mode = &pModeProperties[j];

			if (mode->parameters.visibleRegion.width == width && mode->parameters.visibleRegion.height == height)
			{
				displayMode = mode->displayMode;
				foundMode = true;
				break;
			}
		}
		if (foundMode)
		{
			break;
		}
		delete [] pModeProperties;
	}

	if(!foundMode)
	{
		vks::tools::exitFatal("Can't find a display and a display mode!", -1);
		return;
	}

	// Search for a best plane we can use
	uint32_t bestPlaneIndex = UINT32_MAX;
	VkDisplayKHR* pDisplays = NULL;
	for(uint32_t i = 0; i < planePropertyCount; i++)
	{
		uint32_t planeIndex=i;
		uint32_t displayCount;
		vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, &displayCount, NULL);
		if (pDisplays)
		{
			delete [] pDisplays;
		}
		pDisplays = new VkDisplayKHR[displayCount];
		vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, &displayCount, pDisplays);

		// Find a display that matches the current plane
		bestPlaneIndex = UINT32_MAX;
		for(uint32_t j = 0; j < displayCount; j++)
		{
			if(display == pDisplays[j])
			{
				bestPlaneIndex = i;
				break;
			}
		}
		if(bestPlaneIndex != UINT32_MAX)
		{
			break;
		}
	}

	if(bestPlaneIndex == UINT32_MAX)
	{
		vks::tools::exitFatal("Can't find a plane for displaying!", -1);
		return;
	}

	VkDisplayPlaneCapabilitiesKHR planeCap;
	vkGetDisplayPlaneCapabilitiesKHR(physicalDevice, displayMode, bestPlaneIndex, &planeCap);
	VkDisplayPlaneAlphaFlagBitsKHR alphaMode = (VkDisplayPlaneAlphaFlagBitsKHR)0;

	if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR;
	}
	else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR;
	}
	else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR;
	}
	else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
	}

	VkDisplaySurfaceCreateInfoKHR surfaceInfo{};
	surfaceInfo.sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR;
	surfaceInfo.pNext = NULL;
	surfaceInfo.flags = 0;
	surfaceInfo.displayMode = displayMode;
	surfaceInfo.planeIndex = bestPlaneIndex;
	surfaceInfo.planeStackIndex = pPlaneProperties[bestPlaneIndex].currentStackIndex;
	surfaceInfo.transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	surfaceInfo.globalAlpha = 1.0;
	surfaceInfo.alphaMode = alphaMode;
	surfaceInfo.imageExtent.width = width;
	surfaceInfo.imageExtent.height = height;

	VkResult result = vkCreateDisplayPlaneSurfaceKHR(instance, &surfaceInfo, NULL, &surface);
	if (result !=VK_SUCCESS) {
		vks::tools::exitFatal("Failed to create surface!", result);
	}

	delete[] pDisplays;
	delete[] pModeProperties;
	delete[] pDisplayProperties;
	delete[] pPlaneProperties;
}
#endif 
