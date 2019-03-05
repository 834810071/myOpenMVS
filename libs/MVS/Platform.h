/*
* Platform.h
*/

#ifndef _MVS_PLATFORM_H_
#define _MVS_PLATFORM_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "Camera.h"


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

// a mobile platform with cameras attached to it
class MVS_API Platform
{
public:
	// structure describing a normalized camera mounted on a platform
	typedef CameraIntern Camera;
	typedef CLISTDEF0IDX(Camera,uint32_t) CameraArr;

	// structure describing a pose along the trajectory of a platform
	struct Pose {
		RMatrix R; // platform's rotation matrix
		CMatrix C; // platform's translation vector in the global coordinate system
		#ifdef _USE_BOOST
		template <class Archive>
		void serialize(Archive& ar, const unsigned int /*version*/) {
			ar & R;
			ar & C;
		}
		#endif
	};
	typedef CLISTDEF0IDX(Pose,uint32_t) PoseArr;

public:
	String name; // platform's name
	CameraArr cameras; // cameras mounted on the platform
	PoseArr poses; // trajectory of the platform 平台轨迹

public:
	inline Platform() {}

	Camera GetCamera(uint32_t cameraID, uint32_t poseID) const;

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ar & name;
		ar & cameras;
		ar & poses;
	}
	#endif
};
typedef MVS_API CLISTDEFIDX(Platform,uint32_t) PlatformArr;
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_PLATFORM_H_
