/*
* Platform.cpp
*/

#include "Common.h"
#include "Platform.h"

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

// 返回归一化绝对相机姿态
Platform::Camera Platform::GetCamera(uint32_t cameraID, uint32_t poseID) const
{
	const Camera& camera = cameras[cameraID];
	const Pose& pose = poses[poseID];
	// 将相机的相对姿态添加到平台
	Camera cam;
	cam.K = camera.K;
	cam.R = camera.R*pose.R;
	cam.C = pose.R.t()*camera.C+pose.C;
	return cam;
} // GetCamera
/*----------------------------------------------------------------*/
