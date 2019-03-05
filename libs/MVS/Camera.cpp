/*
* Camera.cpp
*/

#include "Common.h"
#include "Camera.h"

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

const Camera Camera::IDENTITY(Matrix3x3::eye(), Point3(REAL(0), REAL(0), REAL(0)));

Camera::Camera(const Matrix3x4& _P, bool bUpdate/*=true*/)
	:
	P(_P)
{
	if (bUpdate)
		DecomposeP();
} // Camera
Camera::Camera(const Matrix3x3& _R, const Point3& _C, bool bUpdate/*=true*/)
	:
	CameraIntern(_R, _C)
{
	if (bUpdate)
		ComposeP_RC();
} // Camera
Camera::Camera(const Matrix3x3& _K, const Matrix3x3& _R, const Point3& _C, bool bUpdate/*=true*/)
	:
	CameraIntern(_K, _R, _C)
{
	if (bUpdate)
		ComposeP();
} // Camera
/*----------------------------------------------------------------*/


Camera& Camera::operator= (const CameraIntern& camera)
{
	K = camera.K;
	R = camera.R;
	C = camera.C;
	return *this;
}
/*----------------------------------------------------------------*/


void Camera::ComposeP_RC()
{
	AssembleProjectionMatrix(R, C, P);
} // ComposeP_RC
/*----------------------------------------------------------------*/

void Camera::ComposeP()
{
	AssembleProjectionMatrix(K, R, C, P);
} // ComposeP
/*----------------------------------------------------------------*/

void Camera::DecomposeP_RC()
{
	DecomposeProjectionMatrix(P, R, C);
} // DecomposeP_RC
/*----------------------------------------------------------------*/

void Camera::DecomposeP()
{
	DecomposeProjectionMatrix(P, K, R, C);
} // DecomposeP
/*----------------------------------------------------------------*/

void Camera::Transform(const Matrix3x3& _R, const Point3& _t, const REAL& _s)
{
	R = R * _R.t();
	C = _R*C*_s+_t;
	ComposeP();
} // Transform
/*----------------------------------------------------------------*/


REAL Camera::PointDepth(const Point3& X) const
{
	return P(2,0)*X.x + P(2,1)*X.y + P(2,2)*X.z + P(2,3);
} // PointDepth
/*----------------------------------------------------------------*/

bool Camera::IsInFront(const Point3& X) const
{
	return (PointDepth(X)>0);
} // IsInFront
/*----------------------------------------------------------------*/

REAL Camera::DistanceSq(const Point3& X) const
{
	const Point3 ray = X-C;
	return (SQUARE(ray.x)+SQUARE(ray.y)+SQUARE(ray.z));
} // Distance
/*----------------------------------------------------------------*/


// decomposition of projection matrix into KR[I|-C]: internal calibration ([3,3]), rotation ([3,3]) and translation ([3,1])
// (comparable with OpenCV: normalized cv::decomposeProjectionMatrix)
void MVS::DecomposeProjectionMatrix(const PMatrix& P, KMatrix& K, RMatrix& R, CMatrix& C)
{
	// extract camera center as the right null vector of P
	const Vec4 hC(P.RightNullVector());
	C = (const CMatrix&)hC * INVERT(hC[3]);
	// perform RQ decomposition
	const cv::Mat mP(3,4,cv::DataType<REAL>::type,(void*)P.val);
	cv::RQDecomp3x3(mP(cv::Rect(0,0, 3,3)), K, R);
	// normalize calibration matrix
	K *= INVERT(K(2,2));
	// ensure positive focal length
	if (K(0,0) < 0) {
		ASSERT(K(1,1) < 0);
		NEGATE(K(0,0));
		NEGATE(K(1,1));
		NEGATE(K(0,1));
		NEGATE(K(0,2));
		NEGATE(K(1,2));
		(TMatrix<REAL,2,3>&)R *= REAL(-1);
	}
	ASSERT(R.IsValid());
} // DecomposeProjectionMatrix
void MVS::DecomposeProjectionMatrix(const PMatrix& P, RMatrix& R, CMatrix& C)
{
	#ifndef _RELEASE
	KMatrix K;
	DecomposeProjectionMatrix(P, K, R, C);
	ASSERT(K.IsEqual(Matrix3x3::IDENTITY));
	#endif
	// extract camera center as the right null vector of P
	const Vec4 hC(P.RightNullVector());
	C = (const CMatrix&)hC * INVERT(hC[3]);
	// get rotation
	const cv::Mat mP(3,4,cv::DataType<REAL>::type,(void*)P.val);
	mP(cv::Rect(0,0, 3,3)).copyTo(R);
	ASSERT(R.IsValid());
} // DecomposeProjectionMatrix
/*----------------------------------------------------------------*/

// assemble projection matrix: P=KR[I|-C]	组合投影矩阵  空间点映射到图像中 K 3*3	R 3*3 相机旋转矩阵	C 3*1 相机中心
void MVS::AssembleProjectionMatrix(const KMatrix& K, const RMatrix& R, const CMatrix& C, PMatrix& P)
{
	// compute temporary matrices
	cv::Mat mP(3,4,cv::DataType<REAL>::type,(void*)P.val);
	cv::Mat M(mP, cv::Rect(0,0, 3,3));
	cv::Mat(K * R).copyTo(M); //3x3
	mP.col(3) = M * cv::Mat(-C); //3x1	3*3 * 3*1 = 3*1；
} // AssembleProjectionMatrix
void MVS::AssembleProjectionMatrix(const RMatrix& R, const CMatrix& C, PMatrix& P)
{
	Eigen::Map<Matrix3x3::EMat,0,Eigen::Stride<4,0> > eM(P.val);
	eM = (const Matrix3x3::EMat)R;
	Eigen::Map< Point3::EVec,0,Eigen::Stride<0,4> > eT(P.val+3);
	eT = ((const Matrix3x3::EMat)R) * (-((const Point3::EVec)C)); //3x1
} // AssembleProjectionMatrix
/*----------------------------------------------------------------*/
