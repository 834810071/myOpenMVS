/*
* SceneDensify.cpp
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
// MRF: view selection
#include "../Math/TRWS/MRFEnergy.h"
// CGAL: depth-map initialization
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Projection_traits_xy_3.h>

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif


// S T R U C T S ///////////////////////////////////////////////////

// Dense3D data.events
enum EVENT_TYPE {
	EVT_FAIL = 0,
	EVT_CLOSE,

	EVT_PROCESSIMAGE,

	EVT_ESTIMATEDEPTHMAP,
	EVT_OPTIMIZEDEPTHMAP,
	EVT_SAVEDEPTHMAP,

	EVT_FILTERDEPTHMAP,
	EVT_ADJUSTDEPTHMAP,
};

class EVTFail : public Event
{
public:
	EVTFail() : Event(EVT_FAIL) {}
};
class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};

class EVTProcessImage : public Event
{
public:
	IIndex idxImage;
	EVTProcessImage(IIndex _idxImage) : Event(EVT_PROCESSIMAGE), idxImage(_idxImage) {}
};

class EVTEstimateDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTEstimateDepthMap(IIndex _idxImage) : Event(EVT_ESTIMATEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTOptimizeDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTOptimizeDepthMap(IIndex _idxImage) : Event(EVT_OPTIMIZEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTSaveDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTSaveDepthMap(IIndex _idxImage) : Event(EVT_SAVEDEPTHMAP), idxImage(_idxImage) {}
};

class EVTFilterDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTFilterDepthMap(IIndex _idxImage) : Event(EVT_FILTERDEPTHMAP), idxImage(_idxImage) {}
};
class EVTAdjustDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTAdjustDepthMap(IIndex _idxImage) : Event(EVT_ADJUSTDEPTHMAP), idxImage(_idxImage) {}
};
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

// structure used to compute all depth-maps 计算深度映射
class DepthMapsData
{
public:
	DepthMapsData(Scene& _scene);
	~DepthMapsData();

	bool SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap);
	bool SelectViews(DepthData& depthData);
	bool InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors);
	bool InitDepthMap(DepthData& depthData);
	bool EstimateDepthMap(IIndex idxImage);

	bool RemoveSmallSegments(DepthData& depthData);
	bool GapInterpolation(DepthData& depthData);

	bool FilterDepthMap(DepthData& depthData, const IIndexArr& idxNeighbors, bool bAdjust=true);
	void FuseDepthMaps(PointCloud& pointcloud, bool bEstimateNormal);

protected:
	static void* STCALL ScoreDepthMapTmp(void*);
	static void* STCALL EstimateDepthMapTmp(void*);
	static void* STCALL EndDepthMapTmp(void*);

public:
	Scene& scene;

	DepthDataArr arrDepthData;

	// used internally to estimate the depth-maps 内部用于估计深度图
	Image8U::Size prevDepthMapSize; // remember the size of the last estimated depth-map 记住上次估计深度图的大小
	Image8U::Size prevDepthMapSizeTrg; // ... same for target image
	DepthEstimator::MapRefArr coords; // map pixel index to zigzag matrix coordinates 将像素索引映射到锯齿形矩阵坐标
	DepthEstimator::MapRefArr coordsTrg; // ... same for target image
};
/*----------------------------------------------------------------*/


DepthMapsData::DepthMapsData(Scene& _scene)
	:
	scene(_scene),
	arrDepthData(_scene.images.GetSize())
{
} // constructor

DepthMapsData::~DepthMapsData()
{
} // destructor
/*----------------------------------------------------------------*/


// globally choose the best target view for each image,
// trying in the same time the selected image pairs to cover the whole scene;
// the map of selected neighbors for each image is returned in neighborsMap.
// For each view a list of neighbor views ordered by number of shared sparse points and overlapped image area is given.
// Next a graph is formed such that the vertices are the views and two vertices are connected by an edge if the two views have each other as neighbors.
// For each vertex, a list of possible labels is created using the list of neighbor views and scored accordingly (the score is normalized by the average score).
// For each existing edge, the score is defined such that pairing the same two views for any two vertices is discouraged (a constant high penalty is applied for such edges).
// This primal-dual defined problem, even if NP hard, can be solved by a Belief Propagation like algorithm, obtaining in general a solution close enough to optimality.
// 全局选择每个图像的最佳目标视图，     todo
// 同时尝试所选图像对覆盖整个场景；
// 在NeighborsMap中返回每个图像的选定邻居的映射。
// 对于每个视图，给出了按共享稀疏点和重叠图像区域的数量排序的相邻视图的列表。
// 接下来，形成图，使得顶点是视图，并且如果两个视图彼此相邻，则两个顶点通过边连接。
// 对于每个顶点，使用邻居视图列表创建一个可能的标签列表，并相应地打分（分数由平均分数规范化）。
// 对于每个现有边，分数被定义为不鼓励为任何两个顶点配对相同的两个视图（对这样的边应用恒定的高惩罚）。
// 这个原始对偶定义的问题，即使NP困难，也可以用类似信念传播的算法来解决，通常可以得到一个足够接近最优性的解。
bool DepthMapsData::SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap)
{
	// find all pair of images valid for dense reconstruction 查找所有可用于密集重构的图像对
	typedef std::unordered_map<uint64_t,float> PairAreaMap;		// 存储公共区域面积
	PairAreaMap edges;
	double totScore(0);
	unsigned numScores(0);
	// 便利图像和图像对应的邻居，存储其对应的公共区域的图像面积。
	FOREACH(i, images) {
		const IIndex idx(images[i]);
		ASSERT(imagesMap[idx] != NO_ID);
		const ViewScoreArr& neighbors(arrDepthData[idx].neighbors);
		ASSERT(neighbors.GetSize() <= OPTDENSE::nMaxViews);
		// register edges
		FOREACHPTR(pNeighbor, neighbors) {
			const IIndex idx2(pNeighbor->idx.ID);
			ASSERT(imagesMap[idx2] != NO_ID);
			edges[MakePairIdx(idx,idx2)] = pNeighbor->idx.area;
			totScore += pNeighbor->score;
			++numScores;
		}
	}
	if (edges.empty())
		return false;
	const float avgScore((float)(totScore/(double)numScores));

	// run global optimization 执行全局优化  TODO
	const float fPairwiseMul = OPTDENSE::fPairwiseMul; // default 0.3
	const float fEmptyUnaryMult = 6.f;
	const float fEmptyPairwise = 8.f*OPTDENSE::fPairwiseMul;    // 空对
	const float fSamePairwise = 24.f*OPTDENSE::fPairwiseMul;    // 相同对
	const IIndex _num_labels = OPTDENSE::nMaxViews+1; // N neighbors and an empty state
	const IIndex _num_nodes = images.GetSize(); // 输入图像的数量
	typedef MRFEnergy<TypeGeneral> MRFEnergyType;
	CAutoPtr<MRFEnergyType> energy(new MRFEnergyType(TypeGeneral::GlobalSize()));
	CAutoPtrArr<MRFEnergyType::NodeId> nodes(new MRFEnergyType::NodeId[_num_nodes]);
	typedef SEACAVE::cList<TypeGeneral::REAL, const TypeGeneral::REAL&, 0> EnergyCostArr;
	// unary costs: inverse proportional to the image pair score 一元代价:与图像对得分成反比
	EnergyCostArr arrUnary(_num_labels);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		FOREACH(k, neighbors) // 使用平均得分对值进行标准化（不要太依赖于场景中的特征数量）
			arrUnary[k] = avgScore/neighbors[k].score; // use average score to normalize the values (not to depend so much on the number of features in the scene)
		arrUnary[neighbors.GetSize()] = fEmptyUnaryMult*(neighbors.IsEmpty()?avgScore*0.01f:arrUnary[neighbors.GetSize()-1]);
		nodes[n] = energy->AddNode(TypeGeneral::LocalSize(neighbors.GetSize()+1), TypeGeneral::NodeData(arrUnary.Begin()));
	}
	// pairwise costs: as ratios between the area to be covered and the area actually covered
	// 成对费用:按拟覆盖面积与实际覆盖面积之间的比率计算
	EnergyCostArr arrPairwise(_num_labels*_num_labels);
	for (PairAreaMap::const_reference edge: edges) {
		const PairIdx pair(edge.first);
		const float area(edge.second);
		const ViewScoreArr& neighborsI(arrDepthData[pair.i].neighbors);
		const ViewScoreArr& neighborsJ(arrDepthData[pair.j].neighbors);
		arrPairwise.Empty();
		FOREACHPTR(pNj, neighborsJ) {
			const IIndex i(pNj->idx.ID);
			const float areaJ(area/pNj->idx.area);
			FOREACHPTR(pNi, neighborsI) {
				const IIndex j(pNi->idx.ID);
				const float areaI(area/pNi->idx.area);
				arrPairwise.Insert(pair.i == i && pair.j == j ? fSamePairwise : fPairwiseMul*(areaI+areaJ));
			}
			arrPairwise.Insert(fEmptyPairwise+fPairwiseMul*areaJ);
		}
		for (const ViewScore& Ni: neighborsI) {
			const float areaI(area/Ni.idx.area);
			arrPairwise.Insert(fPairwiseMul*areaI+fEmptyPairwise);
		}
		arrPairwise.Insert(fEmptyPairwise*2);
		const IIndex nodeI(imagesMap[pair.i]);
		const IIndex nodeJ(imagesMap[pair.j]);
		energy->AddEdge(nodes[nodeI], nodes[nodeJ], TypeGeneral::EdgeData(TypeGeneral::GENERAL, arrPairwise.Begin()));
	}

	// minimize energy
	MRFEnergyType::Options options;
	options.m_eps = OPTDENSE::fOptimizerEps;
	options.m_iterMax = OPTDENSE::nOptimizerMaxIters;
	#ifndef _RELEASE
	options.m_printIter = 1;
	options.m_printMinIter = 1;
	#endif
	#if 1
	TypeGeneral::REAL energyVal, lowerBound;
	energy->Minimize_TRW_S(options, lowerBound, energyVal);
	#else
	TypeGeneral::REAL energyVal;
	energy->Minimize_BP(options, energyVal);
	#endif

	// extract optimized depth map
	neighborsMap.Resize(_num_nodes);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		IIndex& idxNeighbor = neighborsMap[n];
		const IIndex label((IIndex)energy->GetSolution(nodes[n]));
		ASSERT(label <= neighbors.GetSize());
		if (label == neighbors.GetSize()) {
			idxNeighbor = NO_ID; // empty
		} else {
			idxNeighbor = label;
			DEBUG_ULTIMATE("\treference image %3u paired with target image %3u (idx %2u)", images[n], neighbors[label].idx.ID, label);
		}
	}

	// remove all images with no valid neighbors
	RFOREACH(i, neighborsMap) {
		if (neighborsMap[i] == NO_ID) {
			// remove image with no neighbors
			for (IIndex& imageMap: imagesMap)
				if (imageMap != NO_ID && imageMap > i)
					--imageMap;
			imagesMap[images[i]] = NO_ID;
			images.RemoveAtMove(i);
			neighborsMap.RemoveAtMove(i);
		}
	}
	return !images.IsEmpty();
} // SelectViews
/*----------------------------------------------------------------*/

// compute visibility for the reference image (the first image in "images")
// and select the best views for reconstructing the depth-map;
// extract also all 3D points seen by the reference image
//计算参考图像（“图像”中的第一幅图像）的可见性，并选择重建深度图的最佳视图；
//还提取参考图像所看到的所有3D点  利用Scene.cpp中的查找相邻视图
bool DepthMapsData::SelectViews(DepthData& depthData)
{
	// find and sort valid neighbor views
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(depthData.neighbors.IsEmpty());
	ASSERT(scene.images[idxImage].neighbors.IsEmpty());
	if (!scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews, OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2, FD2R(OPTDENSE::fOptimAngle)))
		return false;
	depthData.neighbors.CopyOf(scene.images[idxImage].neighbors);

	// remove invalid neighbor views
	const float fMinArea(OPTDENSE::fMinArea);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(OPTDENSE::fMinAngle));
	const float fMaxAngle(FD2R(OPTDENSE::fMaxAngle));
	if (!Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, OPTDENSE::nMaxViews)) {
		DEBUG_EXTRA("(libs/MVS/SceneDensify.cpp)error: reference image %3u has no good images in view", idxImage);
		return false;
	}
	return true;
} // SelectViews
/*----------------------------------------------------------------*/

// select target image for the reference image (the first image in "images")
// and initialize images data;
// if idxNeighbor is not NO_ID, only the reference image and the given neighbor are initialized;
// if numNeighbors is not 0, only the first numNeighbors neighbors are initialized;
// otherwise all are initialized;
// returns false if there are no good neighbors to estimate the depth-map
bool DepthMapsData::InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors)
{
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(!depthData.neighbors.IsEmpty());
	ASSERT(depthData.images.IsEmpty());

	// set this image the first image in the array
	depthData.images.Reserve(depthData.neighbors.GetSize()+1);
	depthData.images.AddEmpty();

	if (idxNeighbor != NO_ID) {
		// set target image as the given neighbor
		const ViewScore& neighbor = depthData.neighbors[idxNeighbor];
		DepthData::ViewData& imageTrg = depthData.images.AddEmpty();
		imageTrg.pImageData = &scene.images[neighbor.idx.ID];
		imageTrg.scale = neighbor.idx.scale;
		imageTrg.camera = imageTrg.pImageData->camera;
		imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
		if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
			imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
		DEBUG_EXTRA("Reference image %3u paired with image %3u", idxImage, neighbor.idx.ID);
	} else {
		// init all neighbor views too (global reconstruction is used)
		const float fMinScore(MAXF(depthData.neighbors.First().score*(OPTDENSE::fViewMinScoreRatio*0.1f), OPTDENSE::fViewMinScore));
		FOREACH(idx, depthData.neighbors) {
			const ViewScore& neighbor = depthData.neighbors[idx];
			if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
				(neighbor.score < fMinScore))
				break;
			DepthData::ViewData& imageTrg = depthData.images.AddEmpty();
			imageTrg.pImageData = &scene.images[neighbor.idx.ID];
			imageTrg.scale = neighbor.idx.scale;
			imageTrg.camera = imageTrg.pImageData->camera;
			imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
			if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
				imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// print selected views
		if (g_nVerbosityLevel > 2) {
			String msg;
			for (IDX i=1; i<depthData.images.GetSize(); ++i)
				msg += String::FormatString(" %3u(%.2fscl)", depthData.images[i].pImageData-scene.images.Begin(), depthData.images[i].scale);
			VERBOSE("Reference image %3u paired with %u views:%s (%u shared points)", idxImage, depthData.images.GetSize()-1, msg.c_str(), depthData.points.GetSize());
		} else
		DEBUG_EXTRA("Reference image %3u paired with %u views", idxImage, depthData.images.GetSize()-1);
		#endif
	}
	if (depthData.images.GetSize() < 2) {
		depthData.images.Release();
		return false;
	}

	// init the first image as well
	DepthData::ViewData& imageRef = depthData.images.First();
	imageRef.scale = 1;
	imageRef.pImageData = &scene.images[idxImage];
	imageRef.pImageData->image.toGray(imageRef.image, cv::COLOR_BGR2GRAY, true);
	imageRef.camera = imageRef.pImageData->camera;
	return true;
} // InitViews
/*----------------------------------------------------------------*/

// ???
namespace CGAL {
typedef CGAL::Simple_cartesian<double> kernel_t;
typedef CGAL::Projection_traits_xy_3<kernel_t> Geometry;
typedef CGAL::Delaunay_triangulation_2<Geometry> Delaunay;  // 2维三角网格
typedef CGAL::Delaunay::Face_circulator FaceCirculator;		// 与给定顶点关联的所有面上的循环子
typedef CGAL::Delaunay::Face_handle FaceHandle;
typedef CGAL::Delaunay::Vertex_circulator VertexCirculator;
typedef CGAL::Delaunay::Vertex_handle VertexHandle;			// Vertex_handle可以读取拓展的信息
typedef kernel_t::Point_3 Point;
}

// triangulate in-view points, generating a 2D mesh
// return also the estimated depth boundaries (min and max depth)
// 对视图中的点进行三角剖分，生成二维网格
// 同时返回估计深度边界（最小深度和最大深度）  todo CGAL
std::pair<float,float> TriangulatePointsDelaunay(CGAL::Delaunay& delaunay, const Scene& scene, const DepthData::ViewData& image, const IndexArr& points)
{
	ASSERT(sizeof(Point3) == sizeof(X3D));
	ASSERT(sizeof(Point3) == sizeof(CGAL::Point));
	std::pair<float,float> depthBounds(FLT_MAX, 0.f); // 深度范围 [最大范围， 最小范围]
	// 便利点
	FOREACH(p, points) {    // points里存放的是索引
		const PointCloud::Point& point = scene.pointcloud.points[points[p]];    // 取出scene类中点云中的点
		const Point3 ptCam(image.camera.TransformPointW2C(Cast<REAL>(point)));  // 转换为相机坐标系中  R * (X - C)
		const Point2 ptImg(image.camera.TransformPointC2I(ptCam));              // 相机坐标系转换为像素 [TYPE(K(0,2)+K(0,0)*x.x),TYPE(K(1,2)+K(1,1)*x.y)]
		delaunay.insert(CGAL::Point(ptImg.x, ptImg.y, ptCam.z));                // 三角网格
		const Depth depth((float)ptCam.z);                                      // 深度图
		if (depthBounds.first > depth)
			depthBounds.first = depth;
		if (depthBounds.second < depth)
			depthBounds.second = depth;
	}
	// if full size depth-map requested 如果请求全尺寸深度图
	if (OPTDENSE::bAddCorners) {
		typedef TIndexScore<float,float> DepthDist;	// 深度距离（作为评分）
		typedef CLISTDEF0(DepthDist) DepthDistArr;
		typedef Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::InnerStride<2> > FloatMap;
		// add the four image corners at the average depth 将平均深度处的四个图像角相加
		const CGAL::VertexHandle vcorners[] = {	// 存储4个角的点坐标
			delaunay.insert(CGAL::Point(0, 0, image.pImageData->avgDepth)),	// 左上角
			delaunay.insert(CGAL::Point(image.image.width(), 0, image.pImageData->avgDepth)),	// 右上角
			delaunay.insert(CGAL::Point(0, image.image.height(), image.pImageData->avgDepth)),	// 左下角
			delaunay.insert(CGAL::Point(image.image.width(), image.image.height(), image.pImageData->avgDepth))	// 右下角
		};
		// compute average depth from the closest 3 directly connected faces,
		// weighted by the distance
		// 计算最近的3个直接连接面的平均深度，以距离加权
		const size_t numPoints = 3;
		for (int i=0; i<4; ++i) {
			const CGAL::VertexHandle vcorner = vcorners[i];
			CGAL::FaceCirculator cfc(delaunay.incident_faces(vcorner)); // 入射面
			if (cfc == 0)
				continue; // 正常情况下不应该发生
			const CGAL::FaceCirculator done(cfc);
			Point3d& poszA = (Point3d&)vcorner->point();
			const Point2d& posA = reinterpret_cast<const Point2d&>(poszA);
			const Ray3d rayA(Point3d::ZERO, normalized(image.camera.TransformPointI2C(poszA)));
			DepthDistArr depths(0, numPoints);
			do {
				CGAL::FaceHandle fc(cfc->neighbor(cfc->index(vcorner)));
				if (fc == delaunay.infinite_face())	// 与无限顶点关联的面
					continue;
				for (int j=0; j<4; ++j)
					if (fc->has_vertex(vcorners[j]))
						goto Continue;
				// compute the depth as the intersection of the corner ray with
				// the plane defined by the face's vertices
				// 将深度计算为角射线与面顶点定义的平面的交点
				{
				const Point3d& poszB0 = (const Point3d&)fc->vertex(0)->point();
				const Point3d& poszB1 = (const Point3d&)fc->vertex(1)->point();
				const Point3d& poszB2 = (const Point3d&)fc->vertex(2)->point();
				const Planed planeB(
					image.camera.TransformPointI2C(poszB0),
					image.camera.TransformPointI2C(poszB1),
					image.camera.TransformPointI2C(poszB2)
				);
				const Point3d poszB(rayA.Intersects(planeB));
				if (poszB.z <= 0)
					continue;
				const Point2d posB((reinterpret_cast<const Point2d&>(poszB0)+reinterpret_cast<const Point2d&>(poszB1)+reinterpret_cast<const Point2d&>(poszB2))/3.f);
				const REAL dist(norm(posB-posA));
				depths.StoreTop<numPoints>(DepthDist((float)poszB.z, 1.f/(float)dist));
				}
				Continue:;
			} while (++cfc != done);
			if (depths.GetSize() != numPoints)
				continue; // normally this should never happen
			FloatMap vecDists(&depths[0].score, numPoints);
			vecDists *= 1.f/vecDists.sum();
			FloatMap vecDepths(&depths[0].idx, numPoints);
			poszA.z = vecDepths.dot(vecDists);
		}
	}
	return depthBounds;
}

// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
// 通过对稀疏点云进行三角剖分并对所有像素的法线和深度进行插值，粗略估计深度和法线图
bool DepthMapsData::InitDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.images.First());
	ASSERT(!image.image.empty());   // 图像的浮动强度

	// triangulate in-view points 三角剖分视图中的点
	CGAL::Delaunay delaunay;    // 2维三角网格
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(delaunay, scene, image, depthData.points));	// [大， 小]
	depthData.dMin = thDepth.first*0.9f;
	depthData.dMax = thDepth.second*1.1f;

	// create rough depth-map by interpolating inside triangles 通过内插三角形创建粗略深度图
	const Camera& camera = image.camera;
	depthData.depthMap.create(image.image.size());
	depthData.normalMap.create(image.image.size());
	if (!OPTDENSE::bAddCorners) {
		depthData.depthMap.setTo(Depth(0));
		depthData.normalMap.setTo(0.f);
	}
	struct RasterDepthDataPlaneData { // 光栅深度数据
		const Camera& P;
		DepthMap& depthMap;
		NormalMap& normalMap;
		Point3f normal;
		Point3f normalPlane;
		inline void operator()(const ImageRef& pt) {
			if (!depthMap.isInside(pt))
				return;
			const float z(INVERT(normalPlane.dot(P.TransformPointI2C(Point2f(pt)))));	// 倒置 （）
			ASSERT(z > 0);
			depthMap(pt) = z;
			normalMap(pt) = normal;
		}
	};
	RasterDepthDataPlaneData data = {camera, depthData.depthMap, depthData.normalMap};
	// 便利二维三角网格
	for (CGAL::Delaunay::Face_iterator it=delaunay.faces_begin(); it!=delaunay.faces_end(); ++it) {
		const CGAL::Delaunay::Face& face = *it;
		// 三角网格的三个点
		const Point3f i0((const Point3&)face.vertex(0)->point());
		const Point3f i1((const Point3&)face.vertex(1)->point());
		const Point3f i2((const Point3&)face.vertex(2)->point());
		// 计算由3个点定义的平面
		const Point3f c0(camera.TransformPointI2C(i0));
		const Point3f c1(camera.TransformPointI2C(i1));
		const Point3f c2(camera.TransformPointI2C(i2));
		const Point3f edge1(c1-c0);
		const Point3f edge2(c2-c0);
		data.normal = normalized(edge2.cross(edge1));	// 法线
		data.normalPlane = data.normal * INVERT(data.normal.dot(c0));	// 法平面
		// draw triangle and for each pixel compute depth as the ray intersection with the plane
		// 绘制三角形，并为每个像素计算深度作为光线与平面的交点。
		Image8U::RasterizeTriangle(reinterpret_cast<const Point2f&>(i2), reinterpret_cast<const Point2f&>(i1), reinterpret_cast<const Point2f&>(i0), data);
	}

	DEBUG_ULTIMATE("(libs/MVS/ScenceDensify.cpp)Depth-map %3u roughly estimated from %u sparse points: %dx%d (%s)", &depthData-arrDepthData.Begin(), depthData.points.GetSize(), image.image.width(), image.image.height(), TD_TIMER_GET_FMT().c_str());
	return true;
} // InitDepthMap
/*----------------------------------------------------------------*/


// initialize the confidence map (NCC score map) with the score of the current estimates
// 用当前估计的分数初始化置信度映射（NCC分数映射）
void* STCALL DepthMapsData::ScoreDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) { // 存在置信度都进行遍历
		const ImageRef& x = estimator.coords[idx];	// 参考图像索引
		// 将给定大小的补片居中放置在线段上      获取主图像中的补丁像素值
		if (!estimator.PreparePixelPatch(x) || !estimator.FillPixelPatch()) {
			estimator.depthMap0(x) = 0;
			estimator.normalMap0(x) = Normal::ZERO;
			estimator.confMap0(x) = DepthEstimator::EncodeScoreScale(2.f);
			continue;
		}

		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		const Normal viewDir(Cast<float>(static_cast<const Point3&>(estimator.X0)));	// 法线 ???
		if (depth <= 0) {
			// init with random values 随机数初始化
			depth = DepthEstimator::RandomDepth(estimator.dMin, estimator.dMax);
			normal = DepthEstimator::RandomNormal(viewDir);
		} else if (normal.dot(viewDir) >= 0) {
			// replace invalid normal with random values 用随机值替换无效的法线
			normal = DepthEstimator::RandomNormal(viewDir);
		}
		estimator.confMap0(x) = DepthEstimator::EncodeScoreScale(estimator.ScorePixel(depth, normal));	// 计算NCC分数
	}
	return NULL;
}
// run propagation and random refinement cycles 游程传播和随机精化循环
void* STCALL DepthMapsData::EstimateDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
		estimator.ProcessPixel(idx);
	return NULL;
}
// remove all estimates with too big score and invert confidence map 删除分数太大的所有估计并反转置信度映射
void* STCALL DepthMapsData::EndDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	const float fOptimAngle(FD2R(OPTDENSE::fOptimAngle));
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		float& conf = estimator.confMap0(x);
		const unsigned invScaleRange(DepthEstimator::DecodeScoreScale(conf));	// 解码置信分数
		ASSERT(depth >= 0);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate
		// 检查分数是否足够好，交叉估计是否与当前估计足够接近
		if (conf > OPTDENSE::fNCCThresholdKeep) {	// 最大1 - 比较接受的NCC分数
			#if 1 // used if gap-interpolation is active	如果间隙插值有效，则使用
			conf = 0;
			normal = Normal::ZERO;
			#endif
			depth = 0;
		} else {
			#if 1
			FOREACH(i, estimator.images)
				estimator.scores[i] = ComputeAngle<REAL,float>(estimator.image0.camera.TransformPointI2W(Point3(x,depth)).ptr(), estimator.image0.camera.C.ptr(), estimator.images[i].view.camera.C.ptr());
			#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
			const float fCosAngle(estimator.scores.size() > 1 ? estimator.scores.GetNth(estimator.idxScore) : estimator.scores.front());
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
			const float fCosAngle(estimator.scores.mean());
			#else
			const float fCosAngle(estimator.scores.minCoeff());
			#endif
			const float wAngle(MINF(POW(ACOS(fCosAngle)/fOptimAngle,1.5f),1.f));
			#else
			const float wAngle(1.f);
			#endif
			#if 1
			conf = wAngle/MAXF(conf,1e-2f);
			#elif 1
			conf = wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#else
			conf = SQRT((float)invScaleRange)*wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#endif
		}
	}
	return NULL;
}

// estimate depth-map using propagation and random refinement with NCC score
// as in: "Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes", S. Shen, 2013
// The implementations follows closely the paper, although there are some changes/additions.
// Given two views of the same scene, we note as the "reference image" the view for which a depth-map is reconstructed, and the "target image" the other view.
// As a first step, the whole depth-map is approximated by interpolating between the available sparse points.
// Next, the depth-map is passed from top/left to bottom/right corner and the opposite sens for each of the next steps.
// For each pixel, first the current depth estimate is replaced with its neighbor estimates if the NCC score is better.
// Second, the estimate is refined by trying random estimates around the current depth and normal values, keeping the one with the best score.
// The estimation can be stopped at any point, and usually 2-3 iterations are enough for convergence.
// For each pixel, the depth and normal are scored by computing the NCC score between the patch in the reference image and the wrapped patch in the target image, as dictated by the homography matrix defined by the current values to be estimate.
// In order to ensure some smoothness while locally estimating each pixel, a bonus is added to the NCC score if the estimate for this pixel is close to the estimates for the neighbor pixels.
// Optionally, the occluded pixels can be detected by extending the described iterations to the target image and removing the estimates that do not have similar values in both views.

// 使用传播和随机细化与NCC得分估计深度图
// 如:“使用基于贴片的立体用于大规模场景的精确多视图3D重建”，S.Shen，2013 todo
// 尽管有一些更改/添加，但实现仍然紧跟在本文之后。
// 给定同一场景的两个视图，我们将重建深度图的视图称为“参考图像”，将另一个视图称为“目标图像”。
// 作为第一步，通过在可用稀疏点之间插值来近似整个深度图。
// 接下来，深度图将从上/左到下/右角传递，并在接下来的每个步骤中使用相反的SENS。
// 对于每个像素，如果NCC分数更好，则首先用其邻居估计替换当前深度估计。
// 其次，通过在当前深度和正常值周围尝试随机估计来精化估计值，以保持得分最佳的估计值。
// 估计可以在任何点停止，通常2-3次迭代就足够收敛了。
// 对于每个像素，通过计算参考图像中的贴片和目标图像中的包装贴片之间的NCC得分来对深度和法线进行记分，如由待估计的当前值定义的单应性矩阵所指示的。
// 为了在本地估计每个像素时确保一定的平滑度，如果对该像素的估计接近对相邻像素的估计，则向NCC分数添加奖励。
// 可选地，可以通过将所述迭代扩展到目标图像并移除在两个视图中不具有相似值的估计来检测被遮挡的像素。

bool DepthMapsData::EstimateDepthMap(IIndex idxImage)
{
	TD_TIMER_STARTD();

	// initialize depth and normal maps 初始化深度和法线图
	DepthData& depthData(arrDepthData[idxImage]);
	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.images.First()); // 当前视图
	ASSERT(!image.image.empty() && !depthData.images[1].image.empty()); // （image.image 图像浮动强度 float类型）
	const Image8U::Size size(image.image.size());  // 图像尺寸
	depthData.depthMap.create(size); depthData.depthMap.memset(0);  // 深度图
	depthData.normalMap.create(size);   // 法线图
	depthData.confMap.create(size); // 可信度
	const unsigned nMaxThreads(scene.nMaxThreads);

	// initialize the depth-map 初始化深度图
	if (OPTDENSE::nMinViewsTrustPoint < 2) {    // 校准图像数量
		// compute depth range and initialize known depths
		const int nPixelArea(3); // half windows size around a pixel to be initialize with the known depth 要用已知深度初始化的像素周围的半个窗口大小
		const Camera& camera = depthData.images.First().camera; // 相机矩阵 (K R C P)
		depthData.dMin = FLT_MAX;
		depthData.dMax = 0;
		FOREACHPTR(pPoint, depthData.points) {  // 此图像看到的稀疏3D点的索引
			const PointCloud::Point& X = scene.pointcloud.points[*pPoint];  // 点云中的点坐标
			const Point3 camX(camera.TransformPointW2C(Cast<REAL>(X)));     // 坐标转换为相机坐标系下
			// [ K(0,2)+K(0,0)*x.x, K(1,2)+K(1,1)*x.y ]  偏移值 + 焦距 * 点坐标
			const ImageRef x(ROUND2INT(camera.TransformPointC2I(camX)));    // 从相机空间投影到图像像素
			// ???
			const float d((float)camX.z);   // 深度
			const ImageRef sx(MAXF(x.x-nPixelArea,0), MAXF(x.y-nPixelArea,0)); // 开始
			const ImageRef ex(MINF(x.x+nPixelArea,size.width-1), MINF(x.y+nPixelArea,size.height-1));   // 结束
			for (int y=sx.y; y<=ex.y; ++y)
				for (int x=sx.x; x<=ex.x; ++x)
					depthData.depthMap(y,x) = d;
			if (depthData.dMin > d)
				depthData.dMin = d;
			if (depthData.dMax < d)
				depthData.dMax = d;
		}
		depthData.dMin *= 0.9f;
		depthData.dMax *= 1.1f;
	} else {
		// compute rough estimates using the sparse point-cloud 使用稀疏点云计算粗略估计
		InitDepthMap(depthData);
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// save rough depth map as image 没有使用过
		if (g_nVerbosityLevel > 4) {  // 冗长程度
			ExportDepthMap(ComposeDepthFilePath(idxImage, "init.png"), depthData.depthMap);
			ExportNormalMap(ComposeDepthFilePath(idxImage, "init.normal.png"), depthData.normalMap);
			ExportPointCloud(ComposeDepthFilePath(idxImage, "init.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
		}
		#endif
	}

	// init integral images and index to image-ref map for the reference data
	// 初始化引用数据的图像 引用映射的整数图像和索引
	Image64F imageSum0;
	cv::integral(image.image, imageSum0, CV_64F);	// 计算积分图像
	if (prevDepthMapSize != size) {
		prevDepthMapSize = size;
		BitMatrix mask;
		DepthEstimator::MapMatrix2ZigzagIdx(size, coords, mask, MAXF(64,(int)nMaxThreads*8));	// 队列变为矩阵
	}

	// init threads
	ASSERT(nMaxThreads > 0);
	cList<DepthEstimator> estimators;
	estimators.Reserve(nMaxThreads);
	cList<SEACAVE::Thread> threads;
	if (nMaxThreads > 1)
		threads.Resize(nMaxThreads-1); // current thread is also used
	volatile Thread::safe_t idxPixel;

	// initialize the reference confidence map (NCC score map) with the score of the current estimates
	// 用当前估计的分数初始化参考置信度映射（NCC分数映射）
	{
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(depthData, idxPixel, imageSum0, coords, DepthEstimator::RB2LT);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(ScoreDepthMapTmp, &estimators[i]);
		ScoreDepthMapTmp(&estimators.Last());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// save rough depth map as image 没有用
		if (g_nVerbosityLevel > 4) {
			ExportDepthMap(ComposeDepthFilePath(idxImage, "rough.png"), depthData.depthMap);
			ExportNormalMap(ComposeDepthFilePath(idxImage, "rough.normal.png"), depthData.normalMap);
			ExportPointCloud(ComposeDepthFilePath(idxImage, "rough.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
		}
		#endif
	}

	// run propagation and random refinement cycles on the reference data 参考数据上的运行传播和随机精化循环
	for (unsigned iter=0; iter<OPTDENSE::nEstimationIters; ++iter) {
		// create working threads
		const DepthEstimator::ENDIRECTION dir((DepthEstimator::ENDIRECTION)(iter%DepthEstimator::DIRS));
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(depthData, idxPixel, imageSum0, coords, dir);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(EstimateDepthMapTmp, &estimators[i]);
		EstimateDepthMapTmp(&estimators.Last());
		// 等待所有线程执行结束
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();
		#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
		// save intermediate depth map as image  没用
		if (g_nVerbosityLevel > 4) {
			const String path(ComposeDepthFilePath(image.pImageData-scene.images.Begin(), "iter")+String::ToString(iter));
			ExportDepthMap(path+".png", depthData.depthMap);
			ExportNormalMap(path+".normal.png", depthData.normalMap);
			ExportPointCloud(path+".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
		}
		#endif
	}

	// remove all estimates with too big score and invert confidence map 删除分数太大的所有估计并反转置信度映射
	{
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(depthData, idxPixel, imageSum0, coords, DepthEstimator::DIRS);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(EndDepthMapTmp, &estimators[i]);
		EndDepthMapTmp(&estimators.Last());
		// 等待所有线程执行完毕
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();
	}

	DEBUG_EXTRA("(libs/MVS/SceneDensify.cpp)Depth-map for image %3u %s: %dx%d (%s)", image.pImageData-scene.images.Begin(),
		depthData.images.GetSize() > 2 ?
			String::FormatString("estimated using %2u images", depthData.images.GetSize()-1).c_str() :
			String::FormatString("with image %3u estimated", depthData.images[1].pImageData-scene.images.Begin()).c_str(),
		size.width, size.height, TD_TIMER_GET_FMT().c_str());
	return true;
} // EstimateDepthMap
/*----------------------------------------------------------------*/


// filter out small depth segments from the given depth map  从给定深度图中筛选出小深度段
bool DepthMapsData::RemoveSmallSegments(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*0.7f);	// 细化过程中深度允许的最大差异
	unsigned speckle_size = OPTDENSE::nSpeckleSize;	// 斑点的最大尺寸（小斑点被去除） 100
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	ImageRef size(depthMap.size());

	// 为动态编程数组在堆上分配内存
	TImage<bool> done_map(size, false);
	CAutoPtrArr<ImageRef> seg_list(new ImageRef[size.x*size.y]);
	unsigned seg_list_count;
	unsigned seg_list_curr;
	ImageRef neighbor[4];

	// 遍历所有的像素
	for (int u=0; u<size.x; ++u) {
		for (int v=0; v<size.y; ++v) {
			// 如果此段中的第一个像素已被处理, 跳过
			if (done_map(v,u))
				continue;

			// init segment list (add first element
			// and set it to be the next element to check)
			// 初始化段列表（添加第一个元素并将其设置为下一个要检查的元素）
			seg_list[0] = ImageRef(u,v);	// ImageRef 包含像素的结构体
			seg_list_count = 1;
			seg_list_curr  = 0;

			// add neighboring segments as long as there
			// are none-processed pixels in the seg_list;
			// none-processed means: seg_list_curr<seg_list_count
			// 只要seg_list中有未经处理的像素，就添加相邻的段；
			// 未处理意味着: seg_list_curr<seg_list_count
			while (seg_list_curr < seg_list_count) {
				// get address of current pixel in this segment 获取此段中当前像素的地址
				const ImageRef addr_curr(seg_list[seg_list_curr]);

				// fill list with neighbor positions 用相邻位置填充列表
				neighbor[0] = ImageRef(addr_curr.x-1, addr_curr.y  );
				neighbor[1] = ImageRef(addr_curr.x+1, addr_curr.y  );
				neighbor[2] = ImageRef(addr_curr.x  , addr_curr.y-1);
				neighbor[3] = ImageRef(addr_curr.x  , addr_curr.y+1);

				// 遍历所有的邻居
				const Depth& depth_curr = depthMap(addr_curr);
				for (int i=0; i<4; ++i) {
					// get neighbor pixel address 得到邻居的像素地址
					const ImageRef& addr_neighbor(neighbor[i]);
					// check if neighbor is inside image 检查邻居是否在映像内
					if (addr_neighbor.x>=0 && addr_neighbor.y>=0 && addr_neighbor.x<size.x && addr_neighbor.y<size.y) {
						// check if neighbor has not been added yet 检查邻居是否尚未添加
						bool& done = done_map(addr_neighbor);
						if (!done) {	// 如果没有添加
							// check if the neighbor is valid and similar to the current pixel
							// (belonging to the current segment)
							// 检查邻居是否有效并与当前像素（属于当前段）相似
							const Depth& depth_neighbor = depthMap(addr_neighbor);
							if (depth_neighbor>0 && IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold)) {
								// add neighbor coordinates to segment list 将邻居坐标添加到段列表
								seg_list[seg_list_count++] = addr_neighbor;
								// set neighbor pixel in done_map to "done"
								// (otherwise a pixel may be added 2 times to the list, as
								//  neighbor of one pixel and as neighbor of another pixel)
								//将 done_map 中的相邻像素设置为“done”（否则，一个像素可能作为一个像素的相邻像素和另一个像素的相邻像素被添加到列表中2次）
								done = true;
							}
						}
					}
				}

				// set current pixel in seg_list to "done" 在seg_list中将当前像素设置为"done".
				++seg_list_curr;

				// set current pixel in done_map to "done"	在done_map中将当前像素设置为"done"
				done_map(addr_curr) = true;
			} // end: while (seg_list_curr < seg_list_count)

			// if segment NOT large enough => invalidate pixels
			if (seg_list_count < speckle_size) {	 // 斑点的最大尺寸（小斑点被去除） 100
				// for all pixels in current segment invalidate pixels 对于当前段中的所有像素，使像素无效
				for (unsigned i=0; i<seg_list_count; ++i) {
					depthMap(seg_list[i]) = 0;
					if (!normalMap.empty()) normalMap(seg_list[i]) = Normal::ZERO;
					if (!confMap.empty()) confMap(seg_list[i]) = 0;
				}
			}
		}
	}

	return true;
} // RemoveSmallSegments
/*----------------------------------------------------------------*/

// try to fill small gaps in the depth map 尝试填补深度图中的小空白
bool DepthMapsData::GapInterpolation(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*2.5f);
	unsigned nIpolGapSize = OPTDENSE::nIpolGapSize;	// 插值小间隙（左<->右，顶部<->底部） 7
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	ImageRef size(depthMap.size());

	// 1. Row-wise:  逐行进行
	// for each row do
	for (int v=0; v<size.y; ++v) {
		// init counter 初始化计数器
		unsigned count = 0;

		// for each element of the row do	对于该行的每个元素进行操作
		for (int u=0; u<size.x; ++u) {
			// get depth of this location	// 得到该位置的深度
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it 如果深度无效，跳过
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if speckle is small enough
			// and value in range
			// 检查散斑是否足够小，并且值是否在范围内   当无效的  < 7
			if (count <= nIpolGapSize && (unsigned)u > count) {
				// first value index for interpolation 插值的第一个值索引
				int u_curr(u-count);
				const int u_first(u_curr-1);
				// compute mean depth 计算平均深度
				const Depth& depthFirst = depthMap(v,u_first);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v,u_curr) = avg;
					} while (++u_curr<u);						
					#else
					// interpolate values 插值
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v,u_curr) = (d+=diff);
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v,u_first), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v,u_curr) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v,u_curr));
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					}
					#endif
				}
			}

			// 重置计数器
			count = 0;
		}
	}

	// 2. Column-wise:	逐列进行
	// for each column do 对每列进行操作
	for (int u=0; u<size.x; ++u) {

		// init counter	计数器
		unsigned count = 0;

		// for each element of the column do	对该列的每个元素进行操作
		for (int v=0; v<size.y; ++v) {
			// get depth of this location	获得该位置的深度
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it	如果深度无效，计数并且跳过
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if gap is small enough
			// and value in range
			// 检查散斑是否足够小，并且值是否在范围内   当无效的  < 7
			if (count <= nIpolGapSize && (unsigned)v > count) {
				// first value index for interpolation 插值的第一个值索引
				int v_curr(v-count);
				const int v_first(v_curr-1);
				// compute mean depth 计算评价深度
				const Depth& depthFirst = depthMap(v_first,u);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v_curr,u) = avg;
					} while (++v_curr<v);						
					#else
					// interpolate values 插值
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v_curr,u) = (d+=diff);
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v_first,u), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v_curr,u) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v_curr,u));
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					}
					#endif
				}
			}

			// reset counter 重置计数器
			count = 0;
		}
	}
	return true;
} // GapInterpolation
/*----------------------------------------------------------------*/


// filter depth-map, one pixel at a time, using confidence based fusion or neighbor pixels
// 使用基于置信度的融合或相邻像素一次过滤一个像素的深度图
bool DepthMapsData::FilterDepthMap(DepthData& depthDataRef, const IIndexArr& idxNeighbors, bool 	bAdjust)
{
	TD_TIMER_STARTD();

	// count valid neighbor depth-maps 计算有效邻居深度图
	ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty());
	const IIndex N = idxNeighbors.GetSize();
	ASSERT(OPTDENSE::nMinViewsFilter > 0 && scene.nCalibratedImages > 1);
	// nMinViewsFilter 与估计值一致的最小图像数，以便将其考虑在内
	const IIndex nMinViews(MINF(OPTDENSE::nMinViewsFilter,scene.nCalibratedImages-1));
	// nMinViewsFilterAdjust 符合估计的最小映像数，以便将其视为内部映像（0-禁用）
	const IIndex nMinViewsAdjust(MINF(OPTDENSE::nMinViewsFilterAdjust,scene.nCalibratedImages-1));
	if (N < nMinViews || N < nMinViewsAdjust) {
		DEBUG("(libs/MVS/SceneDensify.cpp)error: depth map %3u can not be filtered", depthDataRef.images.First().pImageData-scene.images.Begin());
		return false;
	}

	// project all neighbor depth-maps to this image 将所有相邻深度图投影到此图像
	const DepthData::ViewData& imageRef = depthDataRef.images.First();	// 参考图像
	const Image8U::Size sizeRef(depthDataRef.depthMap.size());	// 相关图像的数目
	const Camera& cameraRef = imageRef.camera;
	DepthMapArr depthMaps(N);
	ConfidenceMapArr confMaps(N);
	FOREACH(n, depthMaps) {
		DepthMap& depthMap = depthMaps[n];
		depthMap.create(sizeRef);
		depthMap.memset(0);
		ConfidenceMap& confMap = confMaps[n];
		if (bAdjust) {	// 如果需要调整
			confMap.create(sizeRef);
			confMap.memset(0);
		}
		const IIndex idxView = depthDataRef.neighbors[idxNeighbors[(IIndex)n]].idx.ID;
		const DepthData& depthData = arrDepthData[idxView];
		const Camera& camera = depthData.images.First().camera;	// 参考图像对应的相机参数
		const Image8U::Size size(depthData.depthMap.size());
		// 遍历该深度图的每个图像
		for (int i=0; i<size.height; ++i) {
			for (int j=0; j<size.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(depth > 0);
				const Point3 X(camera.TransformPointI2W(Point3(x.x,x.y,depth)));	// 像素坐标系到世界坐标系
				const Point3 camX(cameraRef.TransformPointW2C(X));	// 世界坐标系到相机坐标系
				if (camX.z <= 0)
					continue;
				#if 0
				// set depth on the rounded image projection only
				const ImageRef xRef(ROUND2INT(cameraRef.TransformPointC2I(camX)));
				if (!depthMap.isInside(xRef))
					continue;
				Depth& depthRef(depthMap(xRef));
				if (depthRef != 0 && depthRef < camX.z)
					continue;
				depthRef = camX.z;
				if (bAdjust)
					confMap(xRef) = depthData.confMap(x);
				#else
				// set depth on the 4 pixels around the image projection 设置图像投影周围的4个像素的深度
				const Point2 imgX(cameraRef.TransformPointC2I(camX));
				const ImageRef xRefs[4] = {
					ImageRef(FLOOR2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(FLOOR2INT(imgX.x), CEIL2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), CEIL2INT(imgX.y))
				};
				for (int p=0; p<4; ++p) {
					const ImageRef& xRef = xRefs[p];
					if (!depthMap.isInside(xRef))	// 如果深度图不在参考图像内，则继续
						continue;
					Depth& depthRef(depthMap(xRef));
					if (depthRef != 0 && depthRef < (Depth)camX.z)
						continue;
					depthRef = (Depth)camX.z;
					if (bAdjust)
						confMap(xRef) = depthData.confMap(x);
				}
				#endif
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (g_nVerbosityLevel > 3)
			ExportDepthMap(MAKE_PATH(String::FormatString("depthRender%04u.%04u.png", depthDataRef.images.First().pImageData-scene.images.Begin(), idxView)), depthMap);
		#endif
	}

	// 细化过程中深度允许的最大差异
	const float thDepthDiff(OPTDENSE::fDepthDiffThreshold*1.2f);
	DepthMap newDepthMap(sizeRef);
	ConfidenceMap newConfMap(sizeRef);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	size_t nProcessed(0), nDiscarded(0);
	#endif
	if (bAdjust) {
		// average similar depths, and decrease confidence if depths do not agree 平均相似深度，如果深度不一致，则降低置信度
		// (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// update best depth and confidence estimate with all estimates 用所有估计更新最佳深度和置信度估计
				float posConf(depthDataRef.confMap(xRef)), negConf(0);
				Depth avgDepth(depth*posConf);
				unsigned nPosViews(0), nNegViews(0);
				unsigned n(N);
				do {
					const Depth d(depthMaps[--n](xRef));
					if (d == 0) {
						if (nPosViews + nNegViews + n < nMinViews)
							goto DiscardDepth;
						continue;
					}
					ASSERT(d > 0);
					if (IsDepthSimilar(depth, d, thDepthDiff)) {
						// average similar depths 平均相似深度
						const float c(confMaps[n](xRef));
						avgDepth += d*c;
						posConf += c;
						++nPosViews;
					} else {
						// penalize confidence	降低（损坏）置信度
						if (depth > d) {
							// occlusion 融合
							negConf += confMaps[n](xRef);
						} else {
							// free-space violation  可用空间违规
							const DepthData& depthData = arrDepthData[depthDataRef.neighbors[idxNeighbors[n]].idx.ID];
							const Camera& camera = depthData.images.First().camera;
							const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
							const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
							if (depthData.confMap.isInside(x)) {
								const float c(depthData.confMap(x));
								negConf += (c > 0 ? c : confMaps[n](xRef));
							} else
								negConf += confMaps[n](xRef);
						}
						++nNegViews;
					}
				} while (n);
				ASSERT(nPosViews+nNegViews >= nMinViews);
				// if enough good views and positive confidence... 如果视图和置信度足够好
				if (nPosViews >= nMinViewsAdjust && posConf > negConf && ISINSIDE(avgDepth/=posConf, depthDataRef.dMin, depthDataRef.dMax)) {
					// consider this pixel an inlier 将此像素视为内部像素
					newDepthMap(xRef) = avgDepth;
					newConfMap(xRef) = posConf - negConf;
				} else {
					// consider this pixel an outlier 将此像素视为外部部像素
					DiscardDepth:
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					#if TD_VERBOSE != TD_VERBOSE_OFF
					++nDiscarded;
					#endif
				}
			}
		}
	} else {
		// remove depth if it does not agree with enough neighbors 如果与足够的邻居不一致，则删除深度
		//                           细化过程中深度允许的最大差异 0.01
		const float thDepthDiffStrict(OPTDENSE::fDepthDiffThreshold*0.8f);
		const unsigned nMinGoodViewsProc(75), nMinGoodViewsDeltaProc(65);
		const unsigned nDeltas(4);
		const unsigned nMinViewsDelta(nMinViews*(nDeltas-2));
		const ImageRef xDs[nDeltas] = { ImageRef(-1,0), ImageRef(1,0), ImageRef(0,-1), ImageRef(0,1) };
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// check if very similar with the neighbors projected to this pixel 检查是否与投射到此像素的邻居非常相似
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					unsigned n(N);
					do {
						const Depth d(depthMaps[--n](xRef));
						if (d > 0) {
							// valid view 有效视图
							++nViews;
							if (IsDepthSimilar(depth, d, thDepthDiffStrict)) {
								// agrees with this neighbor 有效视图+1
								++nGoodViews;
							}
						}
					} while (n);
					// 不满足条件
					if (nGoodViews < nMinViews || nGoodViews < nViews*nMinGoodViewsProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// check if similar with the neighbors projected around this pixel 检查是否与此像素周围投影的相邻像素相似
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					for (unsigned d=0; d<nDeltas; ++d) {
						const ImageRef xDRef(xRef+xDs[d]);
						unsigned n(N);
						do {
							const Depth d(depthMaps[--n](xDRef));
							if (d > 0) {
								// valid view 有效视图
								++nViews;
								if (IsDepthSimilar(depth, d, thDepthDiff)) {
									// agrees with this neighbor 有效视图+1
									++nGoodViews;
								}
							}
						} while (n);
					}
					if (nGoodViews < nMinViewsDelta || nGoodViews < nViews*nMinGoodViewsDeltaProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// enough good views, keep it 足够好的视图，保留
				newDepthMap(xRef) = depth;
				newConfMap(xRef) = depthDataRef.confMap(xRef);
			}
		}
	}
	if (!SaveDepthMap(ComposeDepthFilePath(imageRef.pImageData-scene.images.Begin(), "filtered.dmap"), newDepthMap) ||
		!SaveConfidenceMap(ComposeDepthFilePath(imageRef.pImageData-scene.images.Begin(), "filtered.cmap"), newConfMap))
		return false;

	#if TD_VERBOSE != TD_VERBOSE_OFF
	DEBUG("(libs/MVS/SceneDensify.cpp)Depth map %3u filtered using %u other images: %u/%u depths discarded (%s)", imageRef.pImageData-scene.images.Begin(), N, nDiscarded, nProcessed, TD_TIMER_GET_FMT().c_str());
	#endif

	return true;
} // FilterDepthMap
/*----------------------------------------------------------------*/

// fuse all valid depth-maps in the same 3D point cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
// 融合同一3D点云中的所有有效深度图；
// 连接点很可能表示相同的3D点，
// 筛选出阻挡视图的点
struct Proj {
	union {
		uint32_t idxPixel;
		struct {
			uint16_t x, y; // image pixel coordinates
		};
	};
	inline Proj() {}
	inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
	inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
	inline ImageRef GetCoord() const { return ImageRef(x,y); }
};
typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;
// 融合所有深度图
void DepthMapsData::FuseDepthMaps(PointCloud& pointcloud, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	// find best connected images 查找最佳连接图像
	IndexScoreArr connections(0, scene.images.GetSize());
	size_t nPointsEstimate(0);
	FOREACH(i, scene.images) {
		DepthData& depthData = arrDepthData[i];
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(i, "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		IndexScore& connection = connections.AddEmpty();
		connection.idx = i;
		connection.score = (float)scene.images[i].neighbors.GetSize();
		nPointsEstimate += ROUND2INT(depthData.depthMap.area()*(0.5f/*valid*/*0.3f/*new*/));	// 4舍5入
	}
	connections.Sort();

	// fuse all depth-maps, processing the best connected images first 融合所有深度图，首先处理连接最好的图像
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, scene.images.GetSize()));
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.GetSize());
	ProjsArr projs(0, nPointsEstimate);
	// 申请空间
	pointcloud.points.Reserve(nPointsEstimate);
	pointcloud.pointViews.Reserve(nPointsEstimate);
	pointcloud.pointWeights.Reserve(nPointsEstimate);
	Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());	// 进程
	GET_LOGCONSOLE().Pause();
	// 遍历连接
	FOREACHPTR(pConnection, connections) {
		TD_TIMER_STARTD();
		const uint32_t idxImage(pConnection->idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
		for (const ViewScore& neighbor: depthData.neighbors) {
			const Image& imageData = scene.images[neighbor.idx.ID];
			DepthIndex& depthIdxs = arrDepthIdx[&imageData-scene.images.Begin()];
			if (depthIdxs.empty()) {
				depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
				depthIdxs.memset((uint8_t)NO_ID);
			}
		}
		ASSERT(!depthData.IsEmpty());
		const Image8U::Size sizeMap(depthData.depthMap.size());
		const Image& imageData = *depthData.images.First().pImageData;
		ASSERT(&imageData-scene.images.Begin() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const size_t nNumPointsPrev(pointcloud.points.GetSize());
		for (int i=0; i<sizeMap.height; ++i) {
			for (int j=0; j<sizeMap.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = depthIdxs(x);
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point 创建相应的3D点
				idxPoint = (uint32_t)pointcloud.points.GetSize();
				PointCloud::Point& point = pointcloud.points.AddEmpty();
				point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
				PointCloud::ViewArr& views = pointcloud.pointViews.AddEmpty();
				views.Insert(idxImage);
				PointCloud::WeightArr& weights = pointcloud.pointWeights.AddEmpty();
				weights.Insert(depthData.confMap(x));
				ProjArr& pointProjs = projs.AddEmpty();
				pointProjs.Insert(Proj(x));
				// check the projection in the neighbor depth-maps 检查相邻深度图中的投影
				REAL confidence(weights.First());
				Point3 X(point*confidence);
				invalidDepths.Empty();
				FOREACHPTR(pNeighbor, depthData.neighbors) {
					const IIndex idxImageB(pNeighbor->idx.ID);
					const Image& imageDataB = scene.images[idxImageB];
					const Point3f pt(imageDataB.camera.ProjectPointP3(point));
					if (pt.z <= 0)
						continue;
					const ImageRef xB(ROUND2INT(pt.x/pt.z), ROUND2INT(pt.y/pt.z));
					DepthData& depthDataB = arrDepthData[idxImageB];
					DepthMap& depthMapB = depthDataB.depthMap;
					if (!depthMapB.isInside(xB))
						continue;
					Depth& depthB = depthMapB(xB);
					if (depthB == 0)
						continue;
					uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
					if (idxPointB != NO_ID)
						continue;
					if (IsDepthSimilar(pt.z, depthB, OPTDENSE::fDepthDiffThreshold)) {
						// add view to the 3D point
						ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
						const float confidenceB(depthDataB.confMap(xB));
						const IIndex idx(views.InsertSort(idxImageB));
						weights.InsertAt(idx, confidenceB);
						pointProjs.InsertAt(idx, Proj(xB));
						idxPointB = idxPoint;
						X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
						confidence += confidenceB;
					} else
					if (pt.z < depthB) {
						// discard depth  删除深度
						invalidDepths.Insert(&depthB);
					}
				}
				if (views.GetSize() < nMinViewsFuse) {
					// remove point	删除点
					FOREACH(v, views) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(pointProjs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
						arrDepthIdx[idxImageB](x).idx = NO_ID;
					}
					projs.RemoveLast();
					pointcloud.pointWeights.RemoveLast();
					pointcloud.pointViews.RemoveLast();
					pointcloud.points.RemoveLast();
				} else {
					// this point is valid, store it	该点有效，保存它
					point = X*(REAL(1)/confidence);
					ASSERT(ISFINITE(point));
					// invalidate all neighbor depths that do not agree with it 使与其不一致的所有相邻深度无效
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}
		}
		ASSERT(pointcloud.points.GetSize() == pointcloud.pointViews.GetSize() && pointcloud.points.GetSize() == pointcloud.pointWeights.GetSize() && pointcloud.points.GetSize() == projs.GetSize());
		DEBUG_ULTIMATE("(libs/MVS/SceneDensify.cpp)Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.GetSize()-1, pointcloud.points.GetSize()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(pConnection-connections.Begin());
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();

	DEBUG_EXTRA("(libs/MVS/SceneDensify.cpp)Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)", connections.GetSize(), nDepths, pointcloud.points.GetSize(), ROUND2INT((100.f*pointcloud.points.GetSize())/nDepths), TD_TIMER_GET_FMT().c_str());

	if (bEstimateNormal && !pointcloud.points.IsEmpty()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		// 如果需要，也要估算法线（如果法线图不可用，则成本相当高）
		TD_TIMER_STARTD();
		pointcloud.normals.Resize(pointcloud.points.GetSize());
		const int64_t nPoints((int64_t)pointcloud.points.GetSize());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			ASSERT(!weights.IsEmpty());
			IIndex idxView(0);
			float bestWeight = weights.First();
			for (IIndex idx=1; idx<weights.GetSize(); ++idx) {
				const PointCloud::Weight& weight = weights[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.pointViews[i][idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			depthData.GetNormal(projs[i][idxView].GetCoord(), pointcloud.normals[i]);
		}
		DEBUG_EXTRA("(libs/MVS/SceneDensify.cpp)Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.points.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// release all depth-maps
	FOREACHPTR(pDepthData, arrDepthData) {
		if (pDepthData->IsValid())
			pDepthData->DecRef();
	}
} // FuseDepthMaps
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

struct DenseDepthMapData {
	Scene& scene;
	IIndexArr images;
	IIndexArr neighborsMap;
	DepthMapsData detphMaps;
	volatile Thread::safe_t idxImage;	// 易变性
	SEACAVE::EventQueue events; // internal events queue (processed by the working threads) 内设事件队列（按工作流程处理）
	Semaphore sem; // 信号灯 多线程
	CAutoPtr<Util::Progress> progress;

	DenseDepthMapData(Scene& _scene)
		: scene(_scene), detphMaps(_scene), idxImage(0), sem(1) {}

	// 信号完整深度图滤波器
	void SignalCompleteDepthmapFilter() {
		ASSERT(idxImage > 0);
		if (Thread::safeDec(idxImage) == 0)
			sem.Signal((unsigned)images.GetSize()*2);
	}
};

static void* DenseReconstructionEstimateTmp(void*);
static void* DenseReconstructionFilterTmp(void*);

// DensifyPointCloud.cpp的主要算法
// 利用块匹配将稀疏点云变为稠密点云
bool Scene::DenseReconstruction()
{
	DenseDepthMapData data(*this);

	{
	// maps global view indices to our list of views to be processed
	// 将全局视图索引映射到要处理的视图列表
	IIndexArr imagesMap;

	// prepare images for dense reconstruction (load if needed)
	// 为密集重建准备图像（如果需要加载）
	{
		TD_TIMER_START();
		data.images.Reserve(images.GetSize());  // 申请空间
		imagesMap.Resize(images.GetSize());
		#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
		#pragma omp parallel for shared(data, bAbort)   // 对for循环进行并行

		// 从Scene中导入图像
		for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
			#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const IIndex idxImage((IIndex)ID);
		#else
		FOREACH(idxImage, images) {
		#endif
			// skip invalid, uncalibrated or discarded images
			Image& imageData = images[idxImage];
			if (!imageData.IsValid()) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical    // 遇到if定义的情况时，限定以下的部分一次只用一个线程
				#endif
				imagesMap[idxImage] = NO_ID;
				continue;
			}

			// map image index 映射图像索引，即保存到数组当中
			#ifdef DENSE_USE_OPENMP
			#pragma omp critical    // 遇到if定义的情况时，限定以下的部分一次只用一个线程
			#endif
			{
				imagesMap[idxImage] = data.images.GetSize();
				data.images.Insert(idxImage);
			}
			// reload image at the appropriate resolution 以适当的分辨率重新加载图像  重新计算最大分辨率
			const unsigned nMaxResolution(imageData.RecomputeMaxResolution(OPTDENSE::nResolutionLevel, OPTDENSE::nMinResolution));
			if (!imageData.ReloadImage(nMaxResolution)) {   // 根据新的分辨率设置图片大小
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return false;
				#endif
			}
			imageData.UpdateCamera(platforms);  // 因为图片的大小改变，重新计算相机的有关参数（内参矩阵）
			// print image camera
			DEBUG_ULTIMATE("(libs/MVS/SceneDensify.cpp)K%d = \n%s", idxImage, cvMat2String(imageData.camera.K).c_str());
			DEBUG_LEVEL(3, "(libs/MVS/SceneDensify.cpp)R%d = \n%s", idxImage, cvMat2String(imageData.camera.R).c_str());
			DEBUG_LEVEL(3, "(libs/MVS/SceneDensify.cpp)C%d = \n%s", idxImage, cvMat2String(imageData.camera.C).c_str());
		}
		#ifdef DENSE_USE_OPENMP
		if (bAbort || data.images.IsEmpty()) {
		#else
		if (data.images.IsEmpty()) {
		#endif
			VERBOSE("(libs/MVS/SceneDensify.cpp)error: preparing images for dense reconstruction failed (errors loading images)");
			return false;
		}
		VERBOSE("(libs/MVS/SceneDensify.cpp)Preparing images for dense reconstruction completed: %d images (%s)", images.GetSize(), TD_TIMER_GET_FMT().c_str());
	    }

	// select images to be used for dense reconstruction
	{
		TD_TIMER_START();
		// for each image, find all useful neighbor views
		IIndexArr invalidIDs;
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for shared(data, invalidIDs)
		for (int_t ID=0; ID<(int_t)data.images.GetSize(); ++ID) {
			const IIndex idx((IIndex)ID);
		#else
		FOREACH(idx, data.images) {
		#endif
			const IIndex idxImage(data.images[idx]);
			ASSERT(imagesMap[idxImage] != NO_ID);

			DepthData& depthData(data.detphMaps.arrDepthData[idxImage]);
			// 从最开始的深度图中选择有效的深度图
			if (!data.detphMaps.SelectViews(depthData)) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				invalidIDs.InsertSort(idx);
			}
		}
		// DenseDepthMapData移除非法数据
		RFOREACH(i, invalidIDs) {
			const IIndex idx(invalidIDs[i]);
			imagesMap[data.images.Last()] = idx;
			imagesMap[data.images[idx]] = NO_ID;
			data.images.RemoveAt(idx);
		}
		// globally select a target view for each reference image
		// 全局选择每个参考图像的目标视图
		if (OPTDENSE::nNumViews == 1 && !data.detphMaps.SelectViews(data.images, imagesMap, data.neighborsMap)) {
			VERBOSE("(libs/MVS/SceneDensify.cpp)error: no valid images to be dense reconstructed");
			return false;
		}
		ASSERT(!data.images.IsEmpty());
		VERBOSE("(libs/MVS/SceneDensify.cpp)Selecting images for dense reconstruction completed: %d images (%s)", data.images.GetSize(), TD_TIMER_GET_FMT().c_str());
    }
	}

	// initialize the queue of images to be processed
	// 初始化要处理的图像队列
	data.idxImage = 0;
	ASSERT(data.events.IsEmpty());
	data.events.AddEvent(new EVTProcessImage(0));
	// start working threads
	data.progress = new Util::Progress("Estimated depth-maps", data.images.GetSize());
	GET_LOGCONSOLE().Pause();
	if (nMaxThreads > 1) {
		// multi-thread execution 多线程执行
		cList<SEACAVE::Thread> threads(2);
		FOREACHPTR(pThread, threads)
			pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
		FOREACHPTR(pThread, threads)
			pThread->join();
	} else {
		// single-thread execution 单线程执行
		DenseReconstructionEstimate((void*)&data);
	}
	GET_LOGCONSOLE().Play();
	if (!data.events.IsEmpty())
		return false;
	data.progress.Release();

	if ((OPTDENSE::nOptimize & OPTDENSE::ADJUST_FILTER) != 0) {	// 调整滤波器 (1 << 2)
		// initialize the queue of depth-maps to be filtered 初始化要筛选的深度映射队列
		data.sem.Clear();
		data.idxImage = data.images.GetSize();
		ASSERT(data.events.IsEmpty());
		FOREACH(i, data.images)
			data.events.AddEvent(new EVTFilterDepthMap(i));
		// start working threads 开始工作线程
		data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		if (nMaxThreads > 1) {
			// multi-thread execution 多线程
			cList<SEACAVE::Thread> threads(MINF(nMaxThreads, (unsigned)data.images.GetSize()));
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionFilterTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else {
			// single-thread execution 单线程
			DenseReconstructionFilter((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();
	}

	// fuse all depth-maps 融合所有深度图
	pointcloud.Release();
	data.detphMaps.FuseDepthMaps(pointcloud, OPTDENSE::nEstimateNormals == 2);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {	// 没用
		// print number of points with 3+ views 打印具有3个以上视图的点数
		size_t nPoints1m(0), nPoints2(0), nPoints3p(0);
		FOREACHPTR(pViews, pointcloud.pointViews) {
			switch (pViews->GetSize())
			{
			case 0:
			case 1:
				++nPoints1m;
				break;
			case 2:
				++nPoints2;
				break;
			default:
				++nPoints3p;
			}
		}
		VERBOSE("(libs/MVS/SceneDensify.cpp)Dense point-cloud composed of:\n\t%u points with 1- views\n\t%u points with 2 views\n\t%u points with 3+ views", nPoints1m, nPoints2, nPoints3p);
	}
	#endif

	if (!pointcloud.IsEmpty()) {
		if (pointcloud.colors.IsEmpty() && OPTDENSE::nEstimateColors == 1)
			EstimatePointColors(images, pointcloud);
		if (pointcloud.normals.IsEmpty() && OPTDENSE::nEstimateNormals == 1)
			EstimatePointNormals(images, pointcloud);
	}
	return true;
} // DenseReconstructionDepthMap
/*----------------------------------------------------------------*/

void* DenseReconstructionEstimateTmp(void* arg) {
	const DenseDepthMapData& dataThreads = *((const DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionEstimate(arg);
	return NULL;
}

// initialize the dense reconstruction with the sparse point cloud
// 用稀疏点云初始化稠密重建  Important  todo
void Scene::DenseReconstructionEstimate(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	while (true) {
		CAutoPtr<Event> evt(data.events.GetEvent());
		switch (evt->GetID()) {
		case EVT_PROCESSIMAGE: {    // 处理图像
			const EVTProcessImage& evtImage = *((EVTProcessImage*)(Event*)evt); // ???
			if (evtImage.idxImage >= data.images.GetSize()) {   // 如果图像索引大于数量，溢出，退出。
				if (nMaxThreads > 1) {
					// close working threads
					data.events.AddEvent(new EVTClose);
				}
				return;
			}
			// select views to reconstruct the depth-map for this image 选择“视图”以重建此图像的深度图
			const IIndex idx = data.images[evtImage.idxImage];  // 图像索引
			DepthData& depthData(data.detphMaps.arrDepthData[idx]); // 图像深度数据
			// init images pair: reference image and the best neighbor view
			ASSERT(data.neighborsMap.IsEmpty() || data.neighborsMap[evtImage.idxImage] != NO_ID);
			if (!data.detphMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews)) {
				// process next image
				data.events.AddEvent(new EVTProcessImage((IIndex)Thread::safeInc(data.idxImage)));
				break;
			}
			// try to load already compute depth-map for this image 建文件
			if (depthData.Load(ComposeDepthFilePath(idx, "dmap"))) { // 深度图存在就执行优化，否则执行估计。
				if (OPTDENSE::nOptimize & (OPTDENSE::OPTIMIZE)) {
					// optimize depth-map 优化深度图
					data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
				} else {
					// release image data
					depthData.ReleaseImages();
					depthData.Release();
				}
				// process next image
				data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			} else {
				// estimate depth-map 估计深度图
				data.events.AddEventFirst(new EVTEstimateDepthMap(evtImage.idxImage));
			}
			break; }

		case EVT_ESTIMATEDEPTHMAP: {    // 估计深度图
			const EVTEstimateDepthMap& evtImage = *((EVTEstimateDepthMap*)(Event*)evt);
			// request next image initialization to be performed while computing this depth-map 请求在计算此深度图时执行下一图像初始化
			data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			// extract depth map
			data.sem.Wait();
			data.detphMaps.EstimateDepthMap(data.images[evtImage.idxImage]);    // 估计深度图
			data.sem.Signal();	// 多线程
			if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
				// optimize depth-map	优化深度图
				data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
			} else {
				// save depth-map	保存深度图
				data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			}
			break; }

		case EVT_OPTIMIZEDEPTHMAP: {    // 优化深度图
			const EVTOptimizeDepthMap& evtImage = *((EVTOptimizeDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image	将深度图另存为图像
			if (g_nVerbosityLevel > 3) // 冗长程度	未使用
				ExportDepthMap(ComposeDepthFilePath(idx, "raw.png"), depthData.depthMap);
			#endif
			// apply filters 删选
			if (OPTDENSE::nOptimize & (OPTDENSE::REMOVE_SPECKLES)) {	// 去除斑点
				TD_TIMER_START();
				data.detphMaps.RemoveSmallSegments(depthData);
				DEBUG_ULTIMATE("(libs/MVS/SceneDensify.cpp)Depth-map %3u filtered: remove small segments (%s)", idx, TD_TIMER_GET_FMT().c_str());
			}
			if (OPTDENSE::nOptimize & (OPTDENSE::FILL_GAPS)) {	// 间隙插值
				TD_TIMER_START();
				data.detphMaps.GapInterpolation(depthData);
				DEBUG_ULTIMATE("Depth-map %3u filtered: gap interpolation (%s)", idx, TD_TIMER_GET_FMT().c_str());
			}
			// save depth-map
			data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			break; }

		case EVT_SAVEDEPTHMAP: {    // 存档
			const EVTSaveDepthMap& evtImage = *((EVTSaveDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image 保存深度图为图像
			if (g_nVerbosityLevel > 2) {	// 冗长程度	未使用
				ExportDepthMap(ComposeDepthFilePath(idx, "png"), depthData.depthMap);
				ExportConfidenceMap(ComposeDepthFilePath(idx, "conf.png"), depthData.confMap);
				ExportPointCloud(ComposeDepthFilePath(idx, "ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
				if (g_nVerbosityLevel > 4) {
					ExportNormalMap(ComposeDepthFilePath(idx, "normal.png"), depthData.normalMap);
					depthData.confMap.Save(ComposeDepthFilePath(idx, "conf.pfm"));
				}
			}
			#endif
			// save compute depth-map for this image 保存此图像的计算深度图
			depthData.Save(ComposeDepthFilePath(idx, "dmap"));
			depthData.ReleaseImages();
			depthData.Release();
			data.progress->operator++();
			break; }

		case EVT_CLOSE: {
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionEstimate
/*----------------------------------------------------------------*/

void* DenseReconstructionFilterTmp(void* arg) {
	DenseDepthMapData& dataThreads = *((DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionFilter(arg);
	return NULL;
}

// filter estimated depth-maps 筛选估计深度图
void Scene::DenseReconstructionFilter(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	CAutoPtr<Event> evt;
	while ((evt=data.events.GetEvent(0)) != NULL) {
		switch (evt->GetID()) {
		case EVT_FILTERDEPTHMAP: { // 滤波器深度图
			const EVTFilterDepthMap& evtImage = *((EVTFilterDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			if (!depthData.IsValid()) {	// 如果深度数据无效  退出
				data.SignalCompleteDepthmapFilter();
				break;
			}
			// make sure all depth-maps are loaded 确保加载了所有深度图
			depthData.IncRef(ComposeDepthFilePath(idx, "dmap"));	// 加载深度图
			const unsigned numMaxNeighbors(8);	// 最多邻居数量
			IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
			FOREACH(n, depthData.neighbors) {	// 遍历邻居
				const IIndex idxView = depthData.neighbors[n].idx.ID;
				DepthData& depthDataPair = data.detphMaps.arrDepthData[idxView];
				if (!depthDataPair.IsValid())
					continue;
				if (depthDataPair.IncRef(ComposeDepthFilePath(idxView, "dmap")) == 0) {	// 深度图文件读取失败
					// signal error and terminate 信号错误并且终止
					data.events.AddEventFirst(new EVTFail);
					return;
				}
				idxNeighbors.Insert(n);
				if (idxNeighbors.GetSize() == numMaxNeighbors)
					break;
			}
			// filter the depth-map for this image  对于该图像，过滤深度图  在滤波过程中调整深度估计 1
			if (data.detphMaps.FilterDepthMap(depthData, idxNeighbors, OPTDENSE::bFilterAdjust)) {
				// load the filtered maps after all depth-maps were filtered  在筛选所有深度图之后加载筛选的图
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			}
			// unload referenced depth-maps 更新参考深度图
			FOREACHPTR(pIdxNeighbor, idxNeighbors) {
				const IIndex idxView = depthData.neighbors[*pIdxNeighbor].idx.ID;
				DepthData& depthDataPair = data.detphMaps.arrDepthData[idxView];
				depthDataPair.DecRef();	// 减 decrease
			}
			depthData.DecRef();
			data.SignalCompleteDepthmapFilter();	// 多线程相关
			break; }

		case EVT_ADJUSTDEPTHMAP: {	// 调整深度图
			const EVTAdjustDepthMap& evtImage = *((EVTAdjustDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			ASSERT(depthData.IsValid());
			data.sem.Wait();	// 多线程
			// load filtered maps 加载筛选的映射
			if (depthData.IncRef(ComposeDepthFilePath(idx, "dmap")) == 0 ||
				!LoadDepthMap(ComposeDepthFilePath(idx, "filtered.dmap"), depthData.depthMap) ||
				!LoadConfidenceMap(ComposeDepthFilePath(idx, "filtered.cmap"), depthData.confMap))
			{
				// signal error and terminate  信号错误并且终止
				data.events.AddEventFirst(new EVTFail);
				return;
			}
			ASSERT(depthData.GetRef() == 1);
			File::deleteFile(ComposeDepthFilePath(idx, "filtered.dmap").c_str());	// 删除
			File::deleteFile(ComposeDepthFilePath(idx, "filtered.cmap").c_str());
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image 将深度图另存为图像
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(idx, "filtered.png"), depthData.depthMap);
				ExportPointCloud(ComposeDepthFilePath(idx, "filtered.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
			// save filtered depth-map for this image 保存此图像的已筛选深度图
			depthData.Save(ComposeDepthFilePath(idx, "dmap"));
			depthData.DecRef();
			data.progress->operator++();
			break; }

		case EVT_FAIL: {	// 失败
			data.events.AddEventFirst(new EVTFail);
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionFilter
/*----------------------------------------------------------------*/
