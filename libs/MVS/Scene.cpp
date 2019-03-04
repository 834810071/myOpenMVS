/*
* Scene.cpp
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#include "Common.h"
#include "Scene.h"
#define _USE_OPENCV
#include "Interface.h"
#include <iostream>

using namespace std;
using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

#define PROJECT_ID "MVS\0" // identifies the project stream
#define PROJECT_VER ((uint32_t)1) // identifies the version of a project stream


// S T R U C T S ///////////////////////////////////////////////////

void Scene::Release()
{
	platforms.Release();
	images.Release();
	pointcloud.Release();
	mesh.Release();
}

bool Scene::IsEmpty() const
{
	return pointcloud.IsEmpty() && mesh.IsEmpty();
}

// 从接口（Interface类中）提取platforms、image、pointcloud提取到
// platforms(单独Scene类)、image(单独Scene类)、pointcloud(单独Scene类中)
bool Scene::LoadInterface(const String & fileName)
{
	TD_TIMER_STARTD();
	Interface obj;

	// 当前状态序列化
	if (!ARCHIVE::SerializeLoad(obj, fileName))
		return false;

	// 加载 platforms 和 cameras
	ASSERT(!obj.platforms.empty());
	platforms.Reserve((uint32_t)obj.platforms.size());  // 申请空间
	// 遍历平台 将接口(Interface类)中的平台添加到场景(Scene类)中
	for (Interface::PlatformArr::const_iterator itPlatform=obj.platforms.begin(); itPlatform!=obj.platforms.end(); ++itPlatform) {
		Platform& platform = platforms.AddEmpty();  // 先新建一个空的
		platform.name = itPlatform->name;
		platform.cameras.Reserve((uint32_t)itPlatform->cameras.size());	// 平台中的相机申请内存
		// 便利该平台中的相机
		for (Interface::Platform::CameraArr::const_iterator itCamera=itPlatform->cameras.begin(); itCamera!=itPlatform->cameras.end(); ++itCamera) {
			Platform::Camera& camera = platform.cameras.AddEmpty();
			camera.K = itCamera->K;
			camera.R = itCamera->R;
			camera.C = itCamera->C;
			if (!itCamera->IsNormalized()) {
				// 归一化相机内矩阵K
				ASSERT(itCamera->HasResolution());
				const REAL scale(REAL(1)/camera.GetNormalizationScale(itCamera->width,itCamera->height));	// 1 / max（width, height）
				camera.K(0,0) *= scale;
				camera.K(1,1) *= scale;
				camera.K(0,2) *= scale;
				camera.K(1,2) *= scale;
			}
			//DEBUG_EXTRA("Camera model loaded: platform %u; camera %2u; f %.3fx%.3f; poses %u", platforms.GetSize()-1, platform.cameras.GetSize()-1, camera.K(0,0), camera.K(1,1), itPlatform->poses.size());
			DEBUG_EXTRA("（libs/MVS/Scene.cpp）相机模型加载数量: platform(相机平台) %u; camera(相机参数) %2u; f(焦距) %.3f x(偏移量)%.3f; poses(平台轨迹) %u", platforms.GetSize()-1, platform.cameras.GetSize()-1, camera.K(0,0), camera.K(1,1), itPlatform->poses.size());

		}
		ASSERT(platform.cameras.GetSize() == itPlatform->cameras.size());	// 确保相机全部添加进去
		platform.poses.Reserve((uint32_t)itPlatform->poses.size());	// 该平台位姿申请空间
		for (Interface::Platform::PoseArr::const_iterator itPose=itPlatform->poses.begin(); itPose!=itPlatform->poses.end(); ++itPose) {
			Platform::Pose& pose = platform.poses.AddEmpty();
			pose.R = itPose->R;
			pose.C = itPose->C;
		}
		ASSERT(platform.poses.GetSize() == itPlatform->poses.size());
	}
	ASSERT(platforms.GetSize() == obj.platforms.size());
	if (platforms.IsEmpty())
		return false;

	// 导入图像
	nCalibratedImages = 0;	// 多少个标定图像
	size_t nTotalPixels(0);	// 总共像素
	ASSERT(!obj.images.empty());	// 保证图像非空
	images.Reserve((uint32_t)obj.images.size());
	// 遍历图像  image {
	// 				std::string name; // image file name
	//				uint32_t platformID; // ID of the associated platform
	//				uint32_t cameraID; // ID of the associated camera on the associated platform
	//				uint32_t poseID; // ID of the pose of the associated platform
	//				uint32_t ID; // ID of this image in the global space (optional)
	//			}
	// 将接口中的image复制到namespace MVS Image类中
	for (Interface::ImageArr::const_iterator it=obj.images.begin(); it!=obj.images.end(); ++it) {
		const Interface::Image& image = *it;
		const uint32_t ID(images.GetSize());
		Image& imageData = images.AddEmpty();	// 申请空间
		imageData.name = image.name;
		Util::ensureUnifySlash(imageData.name);	// 确保图片名字使用统一斜杠
		imageData.name = MAKE_PATH_FULL(WORKING_FOLDER_FULL, imageData.name);	// 将给定路径添加到给定文件名  WORKING_FOLDER_FULL 当前文件夹的完整路径
		imageData.poseID = image.poseID;
		if (imageData.poseID == NO_ID) {	// 表明未校准
			DEBUG_EXTRA("（libs/MVS/Scene.cpp）warning: uncalibrated image '%s'", image.name.c_str());
			continue;
		}
		imageData.platformID = image.platformID;
		imageData.cameraID = image.cameraID;
		// 初始化 camera
		const Interface::Platform::Camera& camera = obj.platforms[image.platformID].cameras[image.cameraID];
		if (camera.HasResolution()) {
			// 使用存储的像素
			imageData.width = camera.width;
			imageData.height = camera.height;
			imageData.scale = 1;
		} else {
			// 读取图像标题以获得分辨率
			if (!imageData.ReloadImage(0, false))
				return false;
		}
		imageData.UpdateCamera(platforms);  // 从归一化变为非归一化并计算投影矩阵
		++nCalibratedImages;    // 校准图像+1
		nTotalPixels += imageData.width * imageData.height;
		DEBUG_EXTRA("（libs/MVS/Scene.cpp）Image loaded %3u: %s", ID, Util::getFileNameExt(imageData.name).c_str());
	}
	// 如果可以使用的图像数目小于2，则退出
	if (images.GetSize() < 2)
		return false;

	// 加载3D点
	// 将interface类中的3d点复制到Scene类的点云中
	if (!obj.vertices.empty()) {    // 坐标 + 在哪副视图当中
		bool bValidWeights(false);
		pointcloud.points.Resize(obj.vertices.size());  // 点云中的点申请空间
		pointcloud.pointViews.Resize(obj.vertices.size());  // 点云中的视图申请空间
		pointcloud.pointWeights.Resize(obj.vertices.size());	// 点云中点的宽度申请空间
		FOREACH(i, pointcloud.points) {
			const Interface::Vertex& vertex = obj.vertices[i];
			PointCloud::Point& point = pointcloud.points[i];
			point = vertex.X;	// 3D point position

			PointCloud::ViewArr& views = pointcloud.pointViews[i];
			views.Resize((PointCloud::ViewArr::IDX)vertex.views.size());	// 每个点对应的视图数组再申请空间
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			weights.Resize((PointCloud::ViewArr::IDX)vertex.views.size());
			CLISTDEF0(PointCloud::ViewArr::IDX) indices(views.GetSize());	// TODO
			std::iota(indices.Begin(), indices.End(), 0);	// C++中iota是用来批量递增赋值vector的元素；
			std::sort(indices.Begin(), indices.End(), [&](IndexArr::Type i0, IndexArr::Type i1) -> bool {
				return vertex.views[i0].imageID < vertex.views[i1].imageID;
			});
			ASSERT(vertex.views.size() >= 2);
			views.ForEach([&](PointCloud::ViewArr::IDX v) {
				const Interface::Vertex::View& view = vertex.views[indices[v]];
				views[v] = view.imageID;
				weights[v] = view.confidence;
				if (view.confidence != 0)
					bValidWeights = true;
			});
		}
		if (!bValidWeights)	// 如果没有有效的宽度
			pointcloud.pointWeights.Release();
		if (!obj.verticesNormal.empty()) {	// 重建三维点法线数组（可选） 为空
			ASSERT(obj.vertices.size() == obj.verticesNormal.size());
			pointcloud.normals.CopyOf((const Point3f*)&obj.verticesNormal[0].n, obj.vertices.size());	// TODO
		}
		if (!obj.verticesColor.empty()) {
			ASSERT(obj.vertices.size() == obj.verticesColor.size());
			pointcloud.colors.CopyOf((const Pixel8U*)&obj.verticesColor[0].c, obj.vertices.size());
		}
	}

	DEBUG_EXTRA("（libs/MVS/Scene.cpp）Scene loaded from interface format (%s):\n"
				"\t%u images (%u calibrated) with a total of %.2f MPixels (%.2f MPixels/image)\n"
				"\t%u points, %u vertices, %u faces",
				TD_TIMER_GET_FMT().c_str(),
				images.GetSize(), nCalibratedImages, (double)nTotalPixels/(1024.0*1024.0), (double)nTotalPixels/(1024.0*1024.0*nCalibratedImages),
				pointcloud.points.GetSize(), mesh.vertices.GetSize(), mesh.faces.GetSize());
	return true;
} // LoadInterface

// 将platforms(Scene类)、image(Scene类)、pointcloud(Scene类中)保存到接口（Interface类中）
bool Scene::SaveInterface(const String & fileName) const
{
	TD_TIMER_STARTD();
	Interface obj;

	// export platforms	（cameras poses name）
	obj.platforms.reserve(platforms.GetSize());	// 申请空间
	FOREACH(i, platforms) {
		const Platform& platform = platforms[i];
		Interface::Platform plat;
		plat.cameras.reserve(platform.cameras.GetSize());
		// 直接从scene::platform::camera 加入 interface::platform::camera(K[内参矩阵], R[旋转矩阵], C[相机中心]) 数组中
		FOREACH(j, platform.cameras) {
			const Platform::Camera& camera = platform.cameras[j];
			Interface::Platform::Camera cam;
			cam.K = camera.K;
			cam.R = camera.R;
			cam.C = camera.C;
			plat.cameras.push_back(cam);
		}
		plat.poses.reserve(platform.poses.GetSize());
		FOREACH(j, platform.poses) {
			const Platform::Pose& pose = platform.poses[j];
			Interface::Platform::Pose p;
			p.R = pose.R;
			p.C = pose.C;
			plat.poses.push_back(p);
		}
		obj.platforms.push_back(plat);
	}

	// export images
	obj.images.resize(images.GetSize());
	FOREACH(i, images) {
		const Image& imageData = images[i];
		MVS::Interface::Image& image = obj.images[i];
		image.name = MAKE_PATH_REL(WORKING_FOLDER_FULL, imageData.name);	// 仅保存文件名，去除文件路径信息
		image.poseID = imageData.poseID;
		image.platformID = imageData.platformID;
		image.cameraID = imageData.cameraID;
	}

	// export 3D points  （PointArr points;
	//					  PointViewArr pointViews; // array of views for each point (ordered increasing) 每个点的视图数组（按顺序递增）
	//				   	  PointWeightArr pointWeights;
	//					  NormalArr normals;
	//					  ColorArr colors;）
	obj.vertices.resize(pointcloud.points.GetSize());
	FOREACH(i, pointcloud.points) {
		const PointCloud::Point& point = pointcloud.points[i];
		const PointCloud::ViewArr& views = pointcloud.pointViews[i];
		MVS::Interface::Vertex& vertex = obj.vertices[i];
		ASSERT(sizeof(vertex.X.x) == sizeof(point.x));	// 确保空间大小相等
		vertex.X = point;
		vertex.views.resize(views.GetSize());	// interface::vertex::view(imageid, confidence)
		views.ForEach([&](PointCloud::ViewArr::IDX v) {
			MVS::Interface::Vertex::View& view = vertex.views[v];
			view.imageID = views[v];
			view.confidence = (pointcloud.pointWeights.IsEmpty() ? 0.f : pointcloud.pointWeights[i][v]);
		});
	}
	// interface::Normal(一个3d点坐标)
	if (!pointcloud.normals.IsEmpty()) {
		obj.verticesNormal.resize(pointcloud.normals.GetSize());
		FOREACH(i, pointcloud.normals) {
			const PointCloud::Normal& normal = pointcloud.normals[i];
			MVS::Interface::Normal& vertexNormal = obj.verticesNormal[i];
			vertexNormal.n = normal;
		}
	}
	if (!pointcloud.normals.IsEmpty()) {
		obj.verticesColor.resize(pointcloud.colors.GetSize());
		FOREACH(i, pointcloud.colors) {
			const PointCloud::Color& color = pointcloud.colors[i];
			MVS::Interface::Color& vertexColor = obj.verticesColor[i];
			vertexColor.c = color;
		}
	}

	// serialize out the current state 存档
	if (!ARCHIVE::SerializeSave(obj, fileName))
		return false;

	DEBUG_EXTRA("（libs/MVS/Scene.cpp）Scene saved to interface format (%s):\n"
				"\t%u images (%u calibrated)\n"
				"\t%u points, %u vertices, %u faces",
				TD_TIMER_GET_FMT().c_str(),
				images.GetSize(), nCalibratedImages,
				pointcloud.points.GetSize(), mesh.vertices.GetSize(), mesh.faces.GetSize());
	return true;
} // SaveInterface
/*----------------------------------------------------------------*/

// 尝试加载已知的点云或网格文件
bool Scene::Import(const String& fileName)
{
	const String ext(Util::getFileExt(fileName).ToLower());	// 获取文件后缀名
	if (ext == _T(".obj")) {
		// 从obj文件中导入网格
		Release();
		return mesh.Load(fileName);
	}
	if (ext == _T(".ply")) {
		// 从ply文件加载点云/网格
		Release();
		int nVertices(0), nFaces(0);
		{
		PLY ply;
		if (!ply.read(fileName)) {
			DEBUG_EXTRA("（libs/MVS/Scene.cpp）error: invalid PLY file");
			return false;
		}
		for (int i = 0; i < (int)ply.elems.size(); ++i) {
			int elem_count;
			LPCSTR elem_name = ply.setup_element_read(i, &elem_count);
			if (PLY::equal_strings("vertex", elem_name)) {
				nVertices = elem_count;
			} else
			if (PLY::equal_strings("face", elem_name)) {
				nFaces = elem_count;
			}
		}
		}
		if (nVertices && nFaces)
			return mesh.Load(fileName);
		if (nVertices)
			return pointcloud.Load(fileName);
	}
	return false;
} // Import
/*----------------------------------------------------------------*/

// 从稀疏点云文件中加载场景（图片和点云）
bool Scene::Load(const String& fileName, bool bImport)
{
	TD_TIMER_STARTD();
	Release();

	#ifdef _USE_BOOST
	// 打开输入流
	std::ifstream fs(fileName, std::ios::in | std::ios::binary);
	if (!fs.is_open())
		return false;
	// 加载工程头ID
	char szHeader[4];
	fs.read(szHeader, 4);
	/*cout << szHeader << endl;
	cout << PROJECT_ID << endl;
	if (!fs)
	{
		cout << 1 << endl;
	}
	if (_tcsncmp(szHeader, PROJECT_ID, 4) != 0)
	{
		cout << 2 << endl;
	}*/
	// 未进入
	if (!fs || _tcsncmp(szHeader, PROJECT_ID, 4) != 0) {	// 从openMVG的sfm_data.bin转换到openMVS的scene.mvs文件header是"MVSI"，即Interface.h类
		// cout << 3 << endl;
		fs.close();
		if (bImport && Import(fileName))    // 从文件中获取，导入点云或者网格。
			return true;
		if (LoadInterface(fileName))    // 从interface类中获取
			return true;
		VERBOSE("(libs/MVS/Scene.cpp)error: invalid project");
		return false;
	}
	// 加载工程版本
	uint32_t nVer;
	fs.read((char*)&nVer, sizeof(uint32_t));
	if (!fs || nVer != PROJECT_VER) {
		VERBOSE("(libs/MVS/Scene.cpp)error: different project version");
		return false;
	}
	// 加载流类型
	uint32_t nType;
	fs.read((char*)&nType, sizeof(uint32_t));
	// 跳过保留字节
	uint64_t nReserved;
	fs.read((char*)&nReserved, sizeof(uint64_t));
	// 序列化当前状态
	if (!SerializeLoad(*this, fs, (ARCHIVE_TYPE)nType))
		return false;
	// 初始化图像
	nCalibratedImages = 0;
	size_t nTotalPixels(0);
	FOREACH(ID, images) {
		Image& imageData = images[ID];
		if (imageData.poseID == NO_ID)  // 图片必须要有对应的位姿
			continue;
		imageData.UpdateCamera(platforms);  // 从归一化变为非归一化并计算投影矩阵
		++nCalibratedImages;
		nTotalPixels += imageData.width * imageData.height;
	}
	DEBUG_EXTRA("(libs/MVS/Scene.cpp)Scene loaded (%s):\n"
				"\t%u images (%u calibrated) with a total of %.2f MPixels (%.2f MPixels/image)\n"
				"\t%u points, %u vertices, %u faces",
				TD_TIMER_GET_FMT().c_str(),
				images.GetSize(), nCalibratedImages, (double)nTotalPixels/(1024.0*1024.0), (double)nTotalPixels/(1024.0*1024.0*nCalibratedImages),
				pointcloud.points.GetSize(), mesh.vertices.GetSize(), mesh.faces.GetSize());
	return true;
	#else
	return false;
	#endif
} // Load

bool Scene::Save(const String& fileName, ARCHIVE_TYPE type) const
{
	TD_TIMER_STARTD();
	// save using MVS interface if requested    // 如果使用interface类存储，需要保证不是网格类型
	if (type == ARCHIVE_MVS) {
		if (mesh.IsEmpty())
			return SaveInterface(fileName);
		type = ARCHIVE_BINARY_ZIP;
	}
	#ifdef _USE_BOOST
	// open the output stream
	std::ofstream fs(fileName, std::ios::out | std::ios::binary);
	if (!fs.is_open())
		return false;
	// save project ID
	fs.write(PROJECT_ID, 4);
	// save project version
	const uint32_t nVer(PROJECT_VER);
	fs.write((const char*)&nVer, sizeof(uint32_t));
	// save stream type
	const uint32_t nType = type;
	fs.write((const char*)&nType, sizeof(uint32_t));
	// reserve some bytes
	const uint64_t nReserved = 0;
	fs.write((const char*)&nReserved, sizeof(uint64_t));
	// serialize out the current state
	if (!SerializeSave(*this, fs, type))
		return false;
	DEBUG_EXTRA("(libs/MVS/Scene.cpp)Scene saved (%s):\n"
				"\t%u images (%u calibrated)\n"
				"\t%u points, %u vertices, %u faces",
				TD_TIMER_GET_FMT().c_str(),
				images.GetSize(), nCalibratedImages,
				pointcloud.points.GetSize(), mesh.vertices.GetSize(), mesh.faces.GetSize());
	return true;
	#else
	return false;
	#endif
} // Save
/*----------------------------------------------------------------*/


inline float Footprint(const Camera& camera, const Point3f& X) {
	const REAL fSphereRadius(1);    // 球型半径
	const Point3 cX(camera.TransformPointW2C(Cast<REAL>(X)));   // 世界坐标系到相机坐标系
	// TransformPointC2I(从相机空间投影到图像像素 ) x的范数值 （[3, 4] 返回 5）  返回目标数据类型能表示的最逼近1的正数和1的差的绝对值
	return (float)norm(camera.TransformPointC2I(Point3(cX.x+fSphereRadius,cX.y,cX.z))-camera.TransformPointC2I(cX))+std::numeric_limits<float>::epsilon();
}

// 计算参考图像的可见性，并选择重建密集点云的最佳视图；
// 还提取参考图像所看到的所有3D点；
// ("Multi-View Stereo for Community Photo Collections", Goesele, 2007)  TODO                                       10度
bool Scene::SelectNeighborViews(uint32_t ID, IndexArr& points, unsigned nMinViews, unsigned nMinPointViews, float fOptimAngle)
{
	ASSERT(points.IsEmpty());

	// 提取用于参考图像的估计的3D点和对应的2D投影
	Image& imageData = images[ID];
	ASSERT(imageData.IsValid());
	ViewScoreArr& neighbors = imageData.neighbors;  // 记分和存储邻居图像
	ASSERT(neighbors.IsEmpty());
	struct Score {  // 评分
		float score;
		float avgScale;
		float avgAngle;
		uint32_t points;
	};
	CLISTDEF0(Score) scores(images.GetSize());
	scores.Memset(0);
	if (nMinPointViews > nCalibratedImages)
		nMinPointViews = nCalibratedImages;
	unsigned nPoints = 0;
	imageData.avgDepth = 0;
	// 提取参考图像所看到的点
	FOREACH(idx, pointcloud.points) {
		const PointCloud::ViewArr& views = pointcloud.pointViews[idx];  // 该点对应的视图数组
		ASSERT(views.IsSorted());   // 保证视图数组排序
		if (views.FindFirst(ID) == PointCloud::ViewArr::NO_INDEX)   // 如果该视图不存在，则继续 保证参考视图存在
			continue;
		// 存储此点
		const PointCloud::Point& point = pointcloud.points[idx];
		if (views.GetSize() >= nMinPointViews)
			points.Insert((uint32_t)idx);
		imageData.avgDepth += (float)imageData.camera.PointDepth(point);    // P(2,0)*X.x + P(2,1)*X.y + P(2,2)*X.z + P(2,3) P(投影矩阵) TODO
		++nPoints;
		// 对共享视图评分            C 平移（3，1），外部摄像机参数
		const Point3f V1(imageData.camera.C - Cast<REAL>(point));
		const float footprint1(Footprint(imageData.camera, point)); // 足迹
		FOREACHPTR(pView, views) {
			const PointCloud::View& view = *pView;
			if (view == ID) // 如果该视图是当前视图，则继续
				continue;
			const Image& imageData2 = images[view]; // view 是 int类型
			const Point3f V2(imageData2.camera.C - Cast<REAL>(point));
			const float footprint2(Footprint(imageData2.camera, point));
			const float fAngle(ACOS(ComputeAngle<float,float>(V1.ptr(), V2.ptr())));	// 角度
			const float fScaleRatio(footprint1/footprint2);	// 尺度
			const float wAngle(MINF(POW(fAngle/fOptimAngle, 1.5f), 1.f));	// 公式2
			float wScale;
			if (fScaleRatio > 1.6f)
				wScale = SQUARE(1.6f/fScaleRatio);
			else if (fScaleRatio >= 1.f)
				wScale = 1.f;
			else
				wScale = SQUARE(fScaleRatio);	// 幂
			Score& score = scores[view];
			score.score += wAngle * wScale;	// 公式1
			score.avgScale += fScaleRatio;
			score.avgAngle += fAngle;
			++score.points;
		}
	}
	imageData.avgDepth /= nPoints;
	ASSERT(nPoints > 3);

	// 选择最佳 neighborViews
	Point2fArr pointsA(0, points.GetSize()), pointsB(0, points.GetSize());
	FOREACH(IDB, images) {
		const Image& imageDataB = images[IDB];
		if (!imageDataB.IsValid())
			continue;
		const Score& score = scores[IDB];
		if (score.points == 0)
			continue;
		ASSERT(ID != IDB);
		ViewScore& neighbor = neighbors.AddEmpty();
		// 计算匹配特征的分布情况（图像覆盖区域）
		const Point2f boundsA(imageData.GetSize()); // 该视图  图像大小
		const Point2f boundsB(imageDataB.GetSize());
		ASSERT(pointsA.IsEmpty() && pointsB.IsEmpty());
		// 保证点在视图内部
		FOREACHPTR(pIdx, points) {
			const PointCloud::ViewArr& views = pointcloud.pointViews[*pIdx];    // 点对应的视图
			ASSERT(views.IsSorted());
			ASSERT(views.FindFirst(ID) != PointCloud::ViewArr::NO_INDEX);
			if (views.FindFirst(IDB) == PointCloud::ViewArr::NO_INDEX)
				continue;
			const PointCloud::Point& point = pointcloud.points[*pIdx];
			Point2f& ptA = pointsA.AddConstruct(imageData.camera.ProjectPointP(point));
			Point2f& ptB = pointsB.AddConstruct(imageDataB.camera.ProjectPointP(point));
			if (!imageData.camera.IsInside(ptA, boundsA) || !imageDataB.camera.IsInside(ptB, boundsB)) {
				pointsA.RemoveLast();
				pointsB.RemoveLast();
			}
		}
		ASSERT(pointsA.GetSize() == pointsB.GetSize() && pointsA.GetSize() <= score.points);
		const float areaA(ComputeCoveredArea<float, 2, 16, false>((const float*)pointsA.Begin(), pointsA.GetSize(), boundsA.ptr()));
		const float areaB(ComputeCoveredArea<float, 2, 16, false>((const float*)pointsB.Begin(), pointsB.GetSize(), boundsB.ptr()));
		const float area(MINF(areaA, areaB));
		pointsA.Empty(); pointsB.Empty();
		// 存储图像评分
		neighbor.idx.ID = IDB;
		neighbor.idx.points = score.points;
		neighbor.idx.scale = score.avgScale/score.points;
		neighbor.idx.angle = score.avgAngle/score.points;
		neighbor.idx.area = area;
		neighbor.score = score.score*area;
	}
	neighbors.Sort();
	#if TD_VERBOSE != TD_VERBOSE_OFF
	// 打印邻居视图
	if (VERBOSITY_LEVEL > 2) {
		String msg;
		FOREACH(n, neighbors)
			msg += String::FormatString(" %3u(%upts,%.2fscl)", neighbors[n].idx.ID, neighbors[n].idx.points, neighbors[n].idx.scale);
		VERBOSE("(libs/MVS/Scene.cpp)Reference image %3u sees %u views:%s (%u shared points)", ID, neighbors.GetSize(), msg.c_str(), nPoints);
	}
	#endif
	if (points.GetSize() <= 3 || neighbors.GetSize() < MINF(nMinViews,nCalibratedImages-1)) {
		DEBUG_EXTRA("(libs/MVS/Scene.cpp)error: reference image %3u has not enough images in view", ID);
		return false;
	}
	return true;
} // SelectNeighborViews
/*----------------------------------------------------------------*/

// 仅保留参考图像的最佳邻居
bool Scene::FilterNeighborViews(ViewScoreArr& neighbors, float fMinArea, float fMinScale, float fMaxScale, float fMinAngle, float fMaxAngle, unsigned nMaxViews)
{
	// 删除无效的邻居视图
	RFOREACH(n, neighbors) {
		const ViewScore& neighbor = neighbors[n];
		if (neighbor.idx.area < fMinArea ||
			!ISINSIDE(neighbor.idx.scale, fMinScale, fMaxScale) ||
			!ISINSIDE(neighbor.idx.angle, fMinAngle, fMaxAngle))
			neighbors.RemoveAtMove(n);
	}
	if (neighbors.GetSize() > nMaxViews)
		neighbors.Resize(nMaxViews);
	return !neighbors.IsEmpty();
} // FilterNeighborViews
/*----------------------------------------------------------------*/


// export all estimated cameras in a MeshLab MLP project as raster layers
// 导出Meshlab MLP项目中所有估计的摄像机作为光栅层
bool Scene::ExportCamerasMLP(const String& fileName, const String& fileNameScene) const
{
	static const char mlp_header[] =
		"<!DOCTYPE MeshLabDocument>\n"
		"<MeshLabProject>\n"
		" <MeshGroup>\n"
		"  <MLMesh label=\"%s\" filename=\"%s\">\n"
		"   <MLMatrix44>\n"
		"1 0 0 0 \n"
		"0 1 0 0 \n"
		"0 0 1 0 \n"
		"0 0 0 1 \n"
		"   </MLMatrix44>\n"
		"  </MLMesh>\n"
		" </MeshGroup>\n";
	static const char mlp_raster_pos[] =
		"  <MLRaster label=\"%s\">\n"
		"   <VCGCamera TranslationVector=\"%0.6g %0.6g %0.6g 1\"";
	static const char mlp_raster_cam[] =
		" LensDistortion=\"%0.6g %0.6g\""
		" ViewportPx=\"%u %u\""
		" PixelSizeMm=\"1 %0.4f\""
		" FocalMm=\"%0.4f\""
		" CenterPx=\"%0.4f %0.4f\"";
	static const char mlp_raster_rot[] =
		" RotationMatrix=\"%0.6g %0.6g %0.6g 0 %0.6g %0.6g %0.6g 0 %0.6g %0.6g %0.6g 0 0 0 0 1\"/>\n"
		"   <Plane semantic=\"\" fileName=\"%s\"/>\n"
		"  </MLRaster>\n";

	Util::ensureFolder(fileName);
	File f(fileName, File::WRITE, File::CREATE | File::TRUNCATE);

	// write MLP header containing the referenced PLY file
	f.print(mlp_header, Util::getFileName(fileNameScene).c_str(), MAKE_PATH_REL(WORKING_FOLDER_FULL, fileNameScene).c_str());

	// write the raster layers	写入光栅层
	f <<  " <RasterGroup>\n";
	FOREACH(i, images) {
		const Image& imageData = images[i];
		// skip invalid, uncalibrated or discarded images
		if (!imageData.IsValid())
			continue;
		const Camera& camera = imageData.camera;
		f.print(mlp_raster_pos,
			Util::getFileName(imageData.name).c_str(),
			-camera.C.x, -camera.C.y, -camera.C.z
		);
		f.print(mlp_raster_cam,
			0, 0,
			imageData.width, imageData.height,
			camera.K(1,1)/camera.K(0,0), camera.K(0,0),
			camera.K(0,2), camera.K(1,2)
		);
		f.print(mlp_raster_rot,
			 camera.R(0,0),  camera.R(0,1),  camera.R(0,2),
			-camera.R(1,0), -camera.R(1,1), -camera.R(1,2),
			-camera.R(2,0), -camera.R(2,1), -camera.R(2,2),
			MAKE_PATH_REL(WORKING_FOLDER_FULL, imageData.name).c_str()
		);
	}
	f << " </RasterGroup>\n</MeshLabProject>\n";

	return true;
} // ExportCamerasMLP
/*----------------------------------------------------------------*/
