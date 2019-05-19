/*
* SceneTexture.cpp
*/

#include "Common.h"
#include "Scene.h"
#include "RectsBinPack.h"
// connected components
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>
#include <iostream>

using namespace std;
using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define TEXOPT_USE_OPENMP
#endif

// uncomment to use SparseLU for solving the linear systems
// (should be faster, but not working on old Eigen)
#if !defined(EIGEN_DEFAULT_TO_ROW_MAJOR) || EIGEN_WORLD_VERSION>3 || (EIGEN_WORLD_VERSION==3 && EIGEN_MAJOR_VERSION>2)
#define TEXOPT_SOLVER_SPARSELU
#endif

// method used to try to detect outlier face views
// (should enable more consistent textures, but it is not working)
#define TEXOPT_FACEOUTLIER_NA 0
#define TEXOPT_FACEOUTLIER_MEDIAN 1
#define TEXOPT_FACEOUTLIER_GAUSS_DAMPING 2
#define TEXOPT_FACEOUTLIER_GAUSS_CLAMPING 3
#define TEXOPT_FACEOUTLIER TEXOPT_FACEOUTLIER_GAUSS_CLAMPING

// method used to find optimal view per face
#define TEXOPT_INFERENCE_LBP 1
#define TEXOPT_INFERENCE_TRWS 2
#define TEXOPT_INFERENCE TEXOPT_INFERENCE_LBP

// inference algorithm
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
#include "../Math/LBP.h"
namespace MVS {
typedef LBPInference::NodeID NodeID;
// Potts model as smoothness function
LBPInference::EnergyType STCALL SmoothnessPotts(LBPInference::NodeID, LBPInference::NodeID, LBPInference::LabelID l1, LBPInference::LabelID l2) {
	return l1 == l2 && l1 != 0 && l2 != 0 ? LBPInference::EnergyType(0) : LBPInference::EnergyType(LBPInference::MaxEnergy);
}
}
#endif
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
#include "../Math/TRWS/MRFEnergy.h"
namespace MVS {
// TRWS MRF energy using Potts model
typedef unsigned NodeID;
typedef unsigned LabelID;
typedef TypePotts::REAL EnergyType;
static const EnergyType MaxEnergy(1);
struct TRWSInference {
	typedef MRFEnergy<TypePotts> MRFEnergyType;
	typedef MRFEnergy<TypePotts>::Options MRFOptions;

	CAutoPtr<MRFEnergyType> mrf;
	CAutoPtrArr<MRFEnergyType::NodeId> nodes;

	inline TRWSInference() {}
	void Init(NodeID nNodes, LabelID nLabels) {
		mrf = new MRFEnergyType(TypePotts::GlobalSize(nLabels));
		nodes = new MRFEnergyType::NodeId[nNodes];
	}
	inline bool IsEmpty() const {
		return mrf == NULL;
	}
	inline void AddNode(NodeID n, const EnergyType* D) {
		nodes[n] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(D));
	}
	inline void AddEdge(NodeID n1, NodeID n2) {
		mrf->AddEdge(nodes[n1], nodes[n2], TypePotts::EdgeData(MaxEnergy));
	}
	EnergyType Optimize() {
		MRFOptions options;
		options.m_eps = 0.005;
		options.m_iterMax = 1000;
		#if 1
		EnergyType lowerBound, energy;
		mrf->Minimize_TRW_S(options, lowerBound, energy);
		#else
		EnergyType energy;
		mrf->Minimize_BP(options, energy);
		#endif
		return energy;
	}
	inline LabelID GetLabel(NodeID n) const {
		return mrf->GetSolution(nodes[n]);
	}
};
}
#endif

// S T R U C T S ///////////////////////////////////////////////////

typedef Mesh::Vertex Vertex;
typedef Mesh::VIndex VIndex;
typedef Mesh::Face Face;
typedef Mesh::FIndex FIndex;
typedef Mesh::TexCoord TexCoord;

typedef int MatIdx;
typedef Eigen::Triplet<float,MatIdx> MatEntry;
typedef Eigen::SparseMatrix<float,Eigen::ColMajor,MatIdx> SparseMat;

enum Mask {
	empty = 0,
	border = 128,
	interior = 255
};

struct MeshTexture {
	// 用于将表面呈现给视图摄像机
	typedef TImage<cuint32_t> FaceMap;
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		FaceMap& faceMap;
		FIndex idxFace;
		RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap)
			: Base(_vertices, _camera, _depthMap), faceMap(_faceMap) {}
		void Clear() {
			Base::Clear();
			faceMap.memset((uint8_t)NO_ID);
		}
		void Raster(const ImageRef& pt) {
			if (!depthMap.isInside(pt))
				return;
			const Depth z((Depth)INVERT(normalPlane.dot(camera.TransformPointI2C(Point2(pt)))));
			// ASSERT(z > 0);
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				faceMap(pt) = idxFace;
			}
		}
	};

	// used to represent a pixel color
	typedef Point3f Color;
	typedef CLISTDEF0(Color) Colors;

	// used to store info about a face (view, quality)
	struct FaceData {
		VIndex idxView;// the view seeing this face
		float quality; // how well the face is seen by this view
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		Color color; // 另外存储平均颜色（用于移除异常值）
		#endif
	};
	typedef cList<FaceData,const FaceData&,0,8,uint32_t> FaceDataArr; // 存储有关从多个视图中看到的一个网格的信息
	typedef cList<FaceDataArr,const FaceDataArr&,2,1024,FIndex> FaceDataViewArr; // store data for all the faces of the mesh

	// 用来给一个面分配一个视图
	typedef uint32_t Label;
	typedef cList<Label,Label,0,1024,FIndex> LabelArr;

	//代表一个纹理块
	struct TexturePatch {
		Label label; // view index
		Mesh::FaceIdxArr faces; // 贴片包含的面的索引
		RectsBinPack::Rect rect; // 包含修补程序的视图中的边界框。
	};
	typedef cList<TexturePatch,const TexturePatch&,1,1024,FIndex> TexturePatchArr;

	//用于优化纹理贴片
	struct SeamVertex {
		struct Patch {
			struct Edge {
				uint32_t idxSeamVertex; // 这条边的另一个顶点
				FIndex idxFace; // 包含此贴片中此边缘的面

				inline Edge() {}
				inline Edge(uint32_t _idxSeamVertex) : idxSeamVertex(_idxSeamVertex) {}
				inline bool operator == (uint32_t _idxSeamVertex) const {
					return (idxSeamVertex == _idxSeamVertex);
				}
			};
			typedef cList<Edge,const Edge&,0,4,uint32_t> Edges;

			uint32_t idxPatch; // 包含此顶点的贴片
			Point2f proj; // 此顶点在此补丁中的投影
			Edges edges; // 从这个顶点开始的边，包含在这个补丁中（对于流形网格正好是两条）

			inline Patch() {}
			inline Patch(uint32_t _idxPatch) : idxPatch(_idxPatch) {}
			inline bool operator == (uint32_t _idxPatch) const {
				return (idxPatch == _idxPatch);
			}
		};
		typedef cList<Patch,const Patch&,1,4,uint32_t> Patches;

		VIndex idxVertex; // the index of this vertex
		Patches patches; // the patches meeting at this vertex (two or more)

		inline SeamVertex() {}
		inline SeamVertex(uint32_t _idxVertex) : idxVertex(_idxVertex) {}
		inline bool operator == (uint32_t _idxVertex) const {
			return (idxVertex == _idxVertex);
		}
		Patch& GetPatch(uint32_t idxPatch) {
			const uint32_t idx(patches.Find(idxPatch));
			if (idx == NO_ID)
				return patches.AddConstruct(idxPatch);
			return patches[idx];
		}
		inline void SortByPatchIndex(IndexArr& indices) const {
			indices.Resize(patches.GetSize());
			std::iota(indices.Begin(), indices.End(), 0);
			std::sort(indices.Begin(), indices.End(), [&](IndexArr::Type i0, IndexArr::Type i1) -> bool {
				return patches[i0].idxPatch < patches[i1].idxPatch;
			});
		}
	};
	typedef cList<SeamVertex,const SeamVertex&,1,256,uint32_t> SeamVertices;

	// used to iterate vertex labels
	struct PatchIndex {
		bool bIndex;
		union {
			uint32_t idxPatch;
			uint32_t idxSeamVertex;
		};
	};
	typedef CLISTDEF0(PatchIndex) PatchIndices;
	struct VertexPatchIterator {
		uint32_t idx;
		uint32_t idxPatch;
		const SeamVertex::Patches* pPatches;
		inline VertexPatchIterator(const PatchIndex& patchIndex, const SeamVertices& seamVertices) : idx(NO_ID) {
			if (patchIndex.bIndex) {
				pPatches = &seamVertices[patchIndex.idxSeamVertex].patches;
			} else {
				idxPatch = patchIndex.idxPatch;
				pPatches = NULL;
			}
		}
		inline operator uint32_t () const {
			return idxPatch;
		}
		inline bool Next() {
			if (pPatches == NULL)
				return (idx++ == NO_ID);
			if (++idx >= pPatches->GetSize())
				return false;
			idxPatch = (*pPatches)[idx].idxPatch;
			return true;
		}
	};

	// 用于取样接缝边缘
	typedef TAccumulator<Color> AccumColor;	// 对任意类型进行操作的加权累加器类
	typedef Sampler::Linear<float> Sampler;
	struct SampleImage {
		AccumColor accumColor;
		const Image8U3& image;
		const Sampler sampler;

		inline SampleImage(const Image8U3& _image) : image(_image), sampler() {}
		// sample the edge with linear weights
		void AddEdge(const TexCoord& p0, const TexCoord& p1) {
			const TexCoord p01(p1 - p0);
			const float length(norm(p01));
			ASSERT(length > 0.f);
			const int nSamples(ROUND2INT(MAXF(length, 1.f) * 2.f)-1);
			AccumColor edgeAccumColor;
			for (int s=0; s<nSamples; ++s) {
				const float len(static_cast<float>(s) / nSamples);
				const TexCoord samplePos(p0 + p01 * len);
				const Color color(image.sample<Sampler,Color>(sampler, samplePos));
				edgeAccumColor.Add(RGB2YCBCR(color), 1.f-len);
			}
			accumColor.Add(edgeAccumColor.Normalized(), length);
		}
		// returns accumulated color
		Color GetColor() const {
			return accumColor.Normalized();
		}
	};

	// 用于在整个纹理贴片上插值调整颜色
	typedef TImage<Color> ColorMap;
	struct RasterPatchColorData {
		const TexCoord* tri;    // 纹理坐标
		Color colors[3];
		ColorMap& image;

		inline RasterPatchColorData(ColorMap& _image) : image(_image) {}
		inline void operator()(const ImageRef& pt) {
			const Point3f b(BarycentricCoordinates(tri[0], tri[1], tri[2], TexCoord(pt)));
			#if 0
			if (b.x<0 || b.y<0 || b.z<0)
				return; // outside triangle
			#endif
			ASSERT(image.isInside(pt));
			image(pt) = colors[0]*b.x + colors[1]*b.y + colors[2]*b.z;
		}
	};

	// used to compute the coverage of a texture patch
	struct RasterPatchCoverageData {
		const TexCoord* tri;
		Image8U& image;

		inline RasterPatchCoverageData(Image8U& _image) : image(_image) {}
		inline void operator()(const ImageRef& pt) {
			ASSERT(image.isInside(pt));
			image(pt) = interior;
		}
	};

	// used to draw the average edge color of a texture patch
	struct RasterPatchMeanEdgeData {
		Image32F3& image;
		Image8U& mask;
		const Image32F3& image0;
		const Image8U3& image1;
		const TexCoord p0, p0Dir;
		const TexCoord p1, p1Dir;
		const float length;
		const Sampler sampler;

		inline RasterPatchMeanEdgeData(Image32F3& _image, Image8U& _mask, const Image32F3& _image0, const Image8U3& _image1,
									   const TexCoord& _p0, const TexCoord& _p0Adj, const TexCoord& _p1, const TexCoord& _p1Adj)
			: image(_image), mask(_mask), image0(_image0), image1(_image1),
			p0(_p0), p0Dir(_p0Adj-_p0), p1(_p1), p1Dir(_p1Adj-_p1), length((float)norm(p0Dir)), sampler() {}
		inline void operator()(const ImageRef& pt) {
			const float l((float)norm(TexCoord(pt)-p0)/length);
			// compute mean color
			const TexCoord samplePos0(p0 + p0Dir * l);
			AccumColor accumColor(image0.sample<Sampler,Color>(sampler, samplePos0), 1.f);
			const TexCoord samplePos1(p1 + p1Dir * l);
			accumColor.Add(image1.sample<Sampler,Color>(sampler, samplePos1)/255.f, 1.f);
			image(pt) = accumColor.Normalized();
			// set mask edge also
			mask(pt) = border;
		}
	};


public:
	MeshTexture(Scene& _scene, unsigned _nResolutionLevel=0, unsigned _nMinResolution=640);
	~MeshTexture();

	void ListVertexFaces();

	bool ListCameraFaces(FaceDataViewArr&, float fOutlierThreshold);

	#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	bool FaceOutlierDetection(FaceDataArr& faceDatas, float fOutlierThreshold) const;
	#endif

	bool FaceViewSelection(float fOutlierThreshold, float fRatioDataSmoothness);

	void CreateSeamVertices();
	void GlobalSeamLeveling();
	void LocalSeamLeveling();
	void GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty);

	template <typename PIXEL>
	static inline PIXEL RGB2YCBCR(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		return PIXEL(
			v[0] * T(0.299) + v[1] * T(0.587) + v[2] * T(0.114),
			v[0] * T(-0.168736) + v[1] * T(-0.331264) + v[2] * T(0.5) + T(128),
			v[0] * T(0.5) + v[1] * T(-0.418688) + v[2] * T(-0.081312) + T(128)
		);
	}
	template <typename PIXEL>
	static inline PIXEL YCBCR2RGB(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		const T v1(v[1] - T(128));
		const T v2(v[2] - T(128));
		return PIXEL(
			v[0]/* * T(1) + v1 * T(0)*/ + v2 * T(1.402),
			v[0]/* * T(1)*/ + v1 * T(-0.34414) + v2 * T(-0.71414),
			v[0]/* * T(1)*/ + v1 * T(1.772)/* + v2 * T(0)*/
		);
	}


protected:
	static void ProcessMask(Image8U& mask, int stripWidth);
	static void PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias=1.f);


public:
	const unsigned nResolutionLevel; // 在网格优化之前缩小图像多少倍
	const unsigned nMinResolution; // 在网格优化之前缩小图像多少倍

	// store found texture patches
	TexturePatchArr texturePatches;

	// used to compute the seam leveling
	PairIdxArr seamEdges; // 连接不同纹理贴图的（面-面）边缘
	Mesh::FaceIdxArr components; // 对于每个面，存储对应的纹理块索引
	IndexArr mapIdxPatch; // 无效补丁删除后重新映射纹理补丁索引
	SeamVertices seamVertices; // 两个或多个修补程序之间边界上的顶点数组

	//始终有效
	Mesh::VertexFacesArr& vertexFaces; // 对于每个顶点，包含它的面的列表
	BoolArr& vertexBoundary; //对于每个顶点，存储它是否在边界处。
	Mesh::TexCoordArr& faceTexcoords; // 对于每个面，顶点的纹理坐标
	Image8U3& textureDiffuse; //包含漫射颜色的纹理

	// 在整个时间内保持不变
	Mesh::VertexArr& vertices;
	Mesh::FaceArr& faces;
	ImageArr& images;

	Scene& scene; // 网格顶点和面
};

MeshTexture::MeshTexture(Scene& _scene, unsigned _nResolutionLevel, unsigned _nMinResolution)
	:
	nResolutionLevel(_nResolutionLevel),
	nMinResolution(_nMinResolution),
	vertexFaces(_scene.mesh.vertexFaces),
	vertexBoundary(_scene.mesh.vertexBoundary),
	faceTexcoords(_scene.mesh.faceTexcoords),
	textureDiffuse(_scene.mesh.textureDiffuse),
	vertices(_scene.mesh.vertices),
	faces(_scene.mesh.faces),
	images(_scene.images),
	scene(_scene)
{
}
MeshTexture::~MeshTexture()
{
	vertexFaces.Release();
	vertexBoundary.Release();
}

// 提取与每个顶点关联的三角形数组，并检查每个顶点是否在边界处	Mesh.cpp
void MeshTexture::ListVertexFaces()
{
	scene.mesh.EmptyExtra();	// 清空
	// scene.mesh.vertexFaces[face[v]].Insert(i);   包含该顶点的面列表  scene.mesh.vertexFaces[顶点].Insert(面的索引)
	scene.mesh.ListIncidenteFaces();	// 提取与每个顶点关联的三角形数组
	// 当顶点所在的三角形中仅有两个顶点说明该三角形是一条边  属于边界  scene.mesh.vertexBoundary (bool)
	scene.mesh.ListBoundaryVertices();	// 检查每个顶点是否在边界上（确保在前面调用了listIncidenteFaces（））
}

// 提取每个图像看到的面数组                    对faceDatas进行赋值                  6e-2f
// 对应论文 数据项计算 + 图像一致性检查
bool MeshTexture::ListCameraFaces(FaceDataViewArr& facesDatas, float fOutlierThreshold)
{
	// 创建顶点八叉树
	facesDatas.Resize(faces.GetSize());
	typedef std::unordered_set<FIndex> CameraFaces;

	struct FacesInserter {
		FacesInserter(const Mesh::VertexFacesArr& _vertexFaces, CameraFaces& _cameraFaces)
			: vertexFaces(_vertexFaces), cameraFaces(_cameraFaces) {}

        inline void operator() (IDX idxVertex) {
			const Mesh::FaceIdxArr& vertexTris = vertexFaces[idxVertex];
			FOREACHPTR(pTri, vertexTris)
				cameraFaces.emplace(*pTri);
		}

		inline void operator() (const IDX* idices, size_t size) {
			FOREACHRAWPTR(pIdxVertex, idices, size)
				operator()(*pIdxVertex);
		}

		const Mesh::VertexFacesArr& vertexFaces;	// list
		CameraFaces& cameraFaces;	// set
	};

	typedef TOctree<Mesh::VertexArr,float,3> Octree;	// 八叉树类型
	const Octree octree(vertices);	// 顶点构成的八叉树
	#if 0 && !defined(_RELEASE)
	Octree::DEBUGINFO_TYPE info;
	octree.GetDebugInfo(&info);
	Octree::LogDebugInfo(info);
	#endif

	// 提取每个图像看到的面数组
	Util::Progress progress(_T("Initialized views"), images.GetSize());
	typedef float real;
	TImage<real> imageGradMag;  // 梯度幅值
	TImage<real>::EMat mGrad[2];
	FaceMap faceMap;
	DepthMap depthMap;
	#ifdef TEXOPT_USE_OPENMP
	bool bAbort(false);
	#pragma omp parallel for private(imageGradMag, mGrad, faceMap, depthMap)
	// 遍历图像
	for (int_t idx=0; idx<(int_t)images.GetSize(); ++idx) {
		#pragma omp flush (bAbort)
		if (bAbort) {
			++progress;
			continue;
		}
		const uint32_t idxView((uint32_t)idx);  // 图像索引
	#else
	FOREACH(idxView, images) {
	#endif
		Image& imageData = images[idxView];
		if (!imageData.IsValid()) {
			++progress;
			continue;
		}
		// 加载图像
		unsigned level(nResolutionLevel);
		const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));  // 图片大小
		if ((imageData.image.empty() || MAXF(imageData.width,imageData.height) != imageSize) && !imageData.ReloadImage(imageSize)) {
			#ifdef TEXOPT_USE_OPENMP
			bAbort = true;
			#pragma omp flush (bAbort)
			continue;
			#else
			return false;
			#endif
		}

		imageData.UpdateCamera(scene.platforms);    // 更新相机参数

		// 计算梯度幅度
		imageData.image.toGray(imageGradMag, cv::COLOR_BGR2GRAY, true);	// 转换为灰度图像
		cv::Mat grad[2];
		mGrad[0].resize(imageGradMag.rows, imageGradMag.cols);
		grad[0] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[0].data());

		mGrad[1].resize(imageGradMag.rows, imageGradMag.cols);
		grad[1] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[1].data());

		// 使用扩展的Sobel运算符计算第一、第二、第三或混合图像导数。  Sobel 边缘检测算子  论文第6页
		#if 1
		// void Sobel( InputArray src, OutputArray dst, int ddepth,
        //                         int dx, int dy, int ksize = 3,
        //                         double scale = 1, double delta = 0,
        //                         int borderType = BORDER_DEFAULT );
		cv::Sobel(imageGradMag, grad[0], cv::DataType<real>::type, 1, 0, 3, 1.0/8.0);	// Sobel 边缘检测算子 x方向梯度
		cv::Sobel(imageGradMag, grad[1], cv::DataType<real>::type, 0, 1, 3, 1.0/8.0);   // y方向梯度
		#elif 1
		const TMatrix<real,3,5> kernel(CreateDerivativeKernel3x5());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
		#else
		const TMatrix<real,5,7> kernel(CreateDerivativeKernel5x7());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
        #endif
		// (mGrad[0]**2 + mGrad[1]**2).sqrt()
		// (x方向梯度的幂 + y方向梯度的幂)开根号
		(TImage<real>::EMatMap)imageGradMag = (mGrad[0].cwiseAbs2()+mGrad[1].cwiseAbs2()).cwiseSqrt();	// 矩阵和向量的系数方向运算符

		// 选择相机内的面
		CameraFaces cameraFaces;
		FacesInserter inserter(vertexFaces, cameraFaces);
		typedef TFrustum<float,5> Frustum;  //（表示为朝向截头体外部的6个平面） 相机参数
		// 投影矩阵 + 图片 宽 + 图片 高
		const Frustum frustum(Frustum::MATRIX3x4(((PMatrix::CEMatMap)imageData.camera.P).cast<float>()), (float)imageData.width, (float)imageData.height);
		// 遍历树并收集可见索引
		octree.Traverse(frustum, inserter);

		// 投影此视图中的所有三角形并保持最近的三角形
		faceMap.create(imageData.height, imageData.width);
		depthMap.create(imageData.height, imageData.width);

		// 用于投影三角形面
		RasterMesh rasterer(vertices, imageData.camera, depthMap, faceMap); // 光栅网格  点阵数据结构 位图  矩形像素网格
		rasterer.Clear();
		for (auto idxFace : cameraFaces) {
			const Face& facet = faces[idxFace];
			rasterer.idxFace = idxFace;
			rasterer.Project(facet);	// 绘制三角形 投影面顶点到图像平面 得到在图片上三个顶点对应的像素位置
		}

		// 计算投影面积
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		CLISTDEF0IDX(uint32_t,FIndex) areas(faces.GetSize());	// 像素数量  投影区域面积
		areas.Memset(0);
		#endif
		#ifdef TEXOPT_USE_OPENMP
		#pragma omp critical
		#endif
		{
		// 遍历像素点
		// 计算平均颜色 梯度求和  确定faceData对应的视图索引
		for (int j=0; j<faceMap.rows; ++j) {
			for (int i=0; i<faceMap.cols; ++i) {
				const FIndex& idxFace = faceMap(j,i);
				//ASSERT((idxFace == NO_ID && depthMap(j,i) == 0) || (idxFace != NO_ID && depthMap(j,i) > 0));
				if (idxFace == NO_ID)
					continue;

				FaceDataArr& faceDatas = facesDatas[idxFace];   // 对其赋值
				#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				uint32_t& area = areas[idxFace];    // 对面积赋值
				if (area++ == 0) {  // first
				#else
				if (faceDatas.IsEmpty() || faceDatas.Last().idxView != idxView) {
				#endif
					// 创建新的face-data
					FaceData& faceData = faceDatas.AddEmpty();
					faceData.idxView = idxView;
					faceData.quality = imageGradMag(j,i);   // 质量即该像素的梯度赋值
					#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					faceData.color = imageData.image(j,i);	// 图像颜色像素
					#endif
				} else {
					// 更新 face-data
					ASSERT(!faceDatas.IsEmpty());
					FaceData& faceData = faceDatas.Last();
					ASSERT(faceData.idxView == idxView);
					faceData.quality += imageGradMag(j,i);	// 梯度幅度求和 here
					#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					faceData.color += Color(imageData.image(j,i));  // 另外存储平均颜色（用于移除异常值）
					#endif
				}
			}
		}

        // 求颜色均值
        #if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		FOREACH(idxFace, areas) {
			const uint32_t& area = areas[idxFace];
			if (area > 0) {
				Color& color = facesDatas[idxFace].Last().color;
				color = RGB2YCBCR(Color(color * (1.f/(float)area)));    // 像素颜色求取均值
			}
		}
		#endif
		}
		++progress;
	}
	#ifdef TEXOPT_USE_OPENMP
	if (bAbort)
		return false;
	#endif
	progress.close();

	#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	if (fOutlierThreshold > 0) {
		// 尝试检测每个面的异常视图（被场景中的动态对象遮挡的视图，例如行人）
		FOREACHPTR(pFaceDatas, facesDatas)
			FaceOutlierDetection(*pFaceDatas, fOutlierThreshold);   // 图像一致性检查
	}
	#endif
	return true;
}

#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_MEDIAN

// decrease the quality of / remove all views in which the face's projection
// has a much different color than in the majority of views
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	// consider as outlier if the absolute difference to the median is outside this threshold
	if (thOutlier <= 0)
		thOutlier = 0.15f*255.f;

	// init colors array
	if (faceDatas.GetSize() <= 3)
		return false;
	FloatArr channels[3];
	for (int c=0; c<3; ++c)
		channels[c].Resize(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c)
			channels[c][i] = color[c];
	}

	// find median
	for (int c=0; c<3; ++c)
		channels[c].Sort();
	const unsigned idxMedian(faceDatas.GetSize() >> 1);
	Color median;
	for (int c=0; c<3; ++c)
		median[c] = channels[c][idxMedian];

	// abort if there are not at least 3 inliers
	int nInliers(0);
	BoolArr inliers(faceDatas.GetSize());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c) {
			if (ABS(median[c]-color[c]) > thOutlier) {
				inliers[i] = false;
				goto CONTINUE_LOOP;
			}
		}
		inliers[i] = true;
		++nInliers;
		CONTINUE_LOOP:;
	}
	if (nInliers == faceDatas.GetSize())
		return true;
	if (nInliers < 3)
		return false;

	// remove outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	return true;
}

#elif TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA

// 一种多元正态分布，它不是正态分布，积分为1。
// -居中是要使用平均值减去[NX1]来计算函数的向量
// -X是要为其计算函数的向量[NX1]
// -mu是分布以其为中心的平均值[NX1]
// -协方差INV是协方差矩阵[NXN]的逆
// return exp(-1/2 * (X-mu)^T * covariance_inv * (X-mu))
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& centered, const Eigen::Matrix<T,N,N>& covarianceInv) {
    //                  转置矩阵
    return EXP(T(-0.5) * T(centered.adjoint() * covarianceInv * centered)); // 论文 第7页  图像一致性检查
}
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& X, const Eigen::Matrix<T,N,1>& mu, const Eigen::Matrix<T,N,N>& covarianceInv) {
	return MultiGaussUnnormalized<T,N>(X - mu, covarianceInv);
}

// 降低/删除face投影颜色与大多数视图不同的所有视图的质量  图像一致性检查  一个一个进行的
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	// 拒绝所有高斯值低于此阈值的视图
	if (thOutlier <= 0)
		thOutlier = 6e-2f;

	const float minCovariance(1e-3f); // 如果所有协方差都低于此值，则异常值检测将中止

	const unsigned maxIterations(10);   // 最多迭代次数
	const unsigned minInliers(4);   // 内点数小于4 即看到该点的视图数量

	// 初始化颜色(color)数组
	if (faceDatas.GetSize() <= minInliers)
		return false;

	Eigen::Matrix3Xd colorsAll(3, faceDatas.GetSize());	// 三条边 3 * d
	BoolArr inliers(faceDatas.GetSize());
	// 标记为内点  论文中第二步
	FOREACH(i, faceDatas) {
		colorsAll.col(i) = ((const Color::EVec)faceDatas[i].color).cast<double>();
		inliers[i] = true;
	}

	// 执行异常值删除； 如果出现错误（低于阈值的inlier数或无法转换为协方差矩阵），则中止
	size_t numInliers(faceDatas.GetSize());
	Eigen::Vector3d mean;
	Eigen::Matrix3d covariance;
	Eigen::Matrix3d covarianceInv;
	for (unsigned iter = 0; iter < maxIterations; ++iter) {
		// 仅计算内嵌项(inlier)的平均颜色和颜色协方差
		const Eigen::Block<Eigen::Matrix3Xd,3,Eigen::Dynamic,!Eigen::Matrix3Xd::IsRowMajor> colors(colorsAll.leftCols(numInliers));
		mean = colors.rowwise().mean(); // 颜色均值  第一步  mu

		// 协方差(i,j)=（第i列的所有元素-第i列的均值）*（第j列的所有元素-第j列的均值）
		// cov = 1 / (m-1) * x.tranpose() * x   第三步
		const Eigen::Matrix3Xd centered(colors.colwise() - mean);
		covariance = (centered * centered.transpose()) / double(colors.cols() - 1); // 计算协方差  减去均值 除以数量

		// 如果所有协方差变得非常小，则停止
		if (covariance.array().abs().maxCoeff() < minCovariance) {
			// 去除outliers
			RFOREACH(i, faceDatas)
				if (!inliers[i])
					faceDatas.RemoveAt(i);
			return true;
		}

		// 协方差矩阵倒数 （FullPivLU不是最快的，但在倒数期间给出数值稳定性的反馈）
		// 将一个矩阵分解为一个单位下三角矩阵和一个上三角矩阵的乘积
		const Eigen::FullPivLU<Eigen::Matrix3d> lu(covariance); // LU分解的矩阵具有完全旋转性，以及相关的特征。

		// 如果矩阵*的LU分解是可逆的，则为true。
		if (!lu.isInvertible())
			return false;
		covarianceInv = lu.inverse();   // 协方差矩阵的逆
//        cout << covariance << endl;
//        cout << lu << endl;
//        cout << covarianceInv << endl;

        // 筛选内点（高斯值高于阈值的所有视图） 遍历网格数据
		numInliers = 0;
		bool bChanged(false);
		FOREACH(i, faceDatas) {
			const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
			// color -> 内点 mean -> all
			// EXP(T(-0.5) * T(centered.adjoint() * covarianceInv * centered));  第四步
			const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));  // 求取高斯值

			// 阀值比较
			bool& inlier = inliers[i];
			if (gaussValue > thOutlier) {
				// 设置为inlier
				colorsAll.col(numInliers++) = color;
				if (inlier != true) {
					inlier = true;
					bChanged = true;
				}
			} else {
				// 设置为outlier
				if (inlier != false) {
					inlier = false;
					bChanged = true;
				}
			}
		}
		if (numInliers == faceDatas.GetSize())
			return true;
		if (numInliers < minInliers)
			return false;
		if (!bChanged)
			break;
	}

	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_DAMPING
	// select the final inliers
	const float factorOutlierRemoval(0.2f);
	covarianceInv *= factorOutlierRemoval;
	RFOREACH(i, faceDatas) {
		const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
		const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));
		ASSERT(gaussValue >= 0 && gaussValue <= 1);
		faceDatas[i].quality *= gaussValue;
	}
	#endif
	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_CLAMPING
	// 删除outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	#endif
	return true;
}
#endif
//     视图选择                            6e-2f                    0.1
bool MeshTexture::FaceViewSelection(float fOutlierThreshold, float fRatioDataSmoothness)
{
	// 提取与每个顶点关联的三角形数组 并判断该顶点是否在边缘处
	ListVertexFaces();

	// 创建纹理贴图
	{
		// 列出每个面对应的所有视图
		FaceDataViewArr facesDatas; // 数据项

		// 提取每个图像看到的面数组
		// 数据项 + 图像一致性检查
		if (!ListCameraFaces(facesDatas, fOutlierThreshold))	// fOutlierThreshold default(6e-2f)  对facesDatas进行赋值
			return false;

		// 创建面  图  图割
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
		typedef boost::graph_traits<Graph>::edge_iterator EdgeIter;
		typedef boost::graph_traits<Graph>::out_edge_iterator EdgeOutIter;
		Graph graph;    // 图的主要作用是求联通分量
		{
		    // 图中添加顶点  face作为顶点
			FOREACH(idxFace, faces) {
				const Mesh::FIndex idx((Mesh::FIndex)boost::add_vertex(graph));
				ASSERT(idx == idxFace); // 都被添加到图中
			}

			Mesh::FaceIdxArr afaces;
			// 共用边是边
			FOREACH(idxFace, faces) {
			    // 当前面的三个顶点所在的面，如果有重复的则放入afaces  -- adj邻面
				scene.mesh.GetFaceFaces(idxFace, afaces);   // 三个顶点所在的n个面
				ASSERT(ISINSIDE((int)afaces.GetSize(), 1, 4));	// 1 <=size < 4

				FOREACHPTR(pIdxFace, afaces) {
					const FIndex idxFaceAdj = *pIdxFace;
					if (idxFace >= idxFaceAdj)
						continue;

					// 数据项在进行图像一致性检查时删除了一些不符合条件的网格数据项
					const bool bInvisibleFace(facesDatas[idxFace].IsEmpty());   // 数据项是否为空
					const bool bInvisibleFaceAdj(facesDatas[idxFaceAdj].IsEmpty()); // 数据项是否为空

					if (bInvisibleFace || bInvisibleFaceAdj) {
						if (bInvisibleFace != bInvisibleFaceAdj)	// 一个为空，一个不为空则说明存在缝隙
							seamEdges.AddConstruct(idxFace, idxFaceAdj);    // 连接不同纹理贴图的（面-面）边缘
						continue;
					}

					boost::add_edge(idxFace, idxFaceAdj, graph);	// 两个相邻面都不为空  图中的边是点
				}
				afaces.Empty();	// 清空
			}

			ASSERT((Mesh::FIndex)boost::num_vertices(graph) == faces.GetSize());	// 图中顶点数等于面数
		}

		// 给每个面分配最佳视图  todo
		LabelArr labels(faces.GetSize());
	    // cList<FIndex, FIndex, 0, 8, FIndex>
		components.Resize(faces.GetSize());     // 对于每个面，存储对应的纹理块索引
		{
			// 查找连接的组件  返回联通子图数目
			const FIndex nComponents(boost::connected_components(graph, components.Begin()));   // 使用基于DFS的方法计算无向图的连通分量  得到联通数量

			// 将面ID从全局空间映射到组件空间
			typedef cList<NodeID, NodeID, 0, 128, NodeID> NodeIDs;
			NodeIDs nodeIDs(faces.GetSize());
			NodeIDs sizes(nComponents);     // 每个联通数量的大小
			sizes.Memset(0);
			FOREACH(c, components)
				nodeIDs[c] = sizes[components[c]]++;
//			FOREACH(c, components)
//            {
//			    cout << components[c] << endl;
//                cout << sizes[components[c]] << endl;
//                cout << nodeIDs[c] << endl;
//            }


			// 归一化质量值
			// 找到最大质量 -> 梯度幅值
			float maxQuality(0);
			FOREACHPTR(pFaceDatas, facesDatas) {
				const FaceDataArr& faceDatas = *pFaceDatas;
				FOREACHPTR(pFaceData, faceDatas)
					if (maxQuality < pFaceData->quality)
						maxQuality = pFaceData->quality;
			}

			// 直方图  求梯度幅值的均值
			Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);

			// cout << hist.GetStart() << endl;
			FOREACHPTR(pFaceDatas, facesDatas) {
				const FaceDataArr& faceDatas = *pFaceDatas; // 存储有关从多个视图中看到的一个网格的信息
				FOREACHPTR(pFaceData, faceDatas)
					hist.Add(pFaceData->quality);   // 位计数+1    增加保存此直方图在范围内的值的库的计数，如果不在范围内，则增加下溢/溢出计数
			}
			const float normQuality(hist.GetApproximatePermille(0.95f));  //  均值
			// cout << normQuality << endl; 0

			#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
			// 进行平滑项设置
			// 初始化LBPInference  循环置信传播算法的基本实现  会随着变化而变化  每一个结点的标记即为最优标记，MRF也达到了收敛状态
			// https://github.com/nmoehrle/mvs-texturing
			CLISTDEFIDX(LBPInference,FIndex) inferences(nComponents);   // 以联通图为单位
			{

			    // 置信中添加节点
				FOREACH(s, sizes) { // sizes 联通图数量
					const NodeID numNodes(sizes[s]);    // 某一分量的数量
					ASSERT(numNodes > 0);
					if (numNodes <= 1)
						continue;
					LBPInference& inference = inferences[s];
					inference.SetNumNodes(numNodes);
					// 如果标签一致 置为0 如果不一致 置为1 or 最大值
					inference.SetSmoothCost(SmoothnessPotts);   // 论文 第7页
				}

				// 置信传播添加边
				EdgeOutIter ei, eie;
				FOREACH(f, faces) {
					LBPInference& inference = inferences[components[f]];

					for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
						ASSERT(f == (FIndex)ei->m_source);
						const FIndex fAdj((FIndex)ei->m_target);
						ASSERT(components[f] == components[fAdj]);
						if (f < fAdj) //
							inference.SetNeighbors(nodeIDs[f], nodeIDs[fAdj]);
					}
				}
			}

			// 设置数据成本
			{
			                                            // 0.1 * 1000 = 100
				const LBPInference::EnergyType MaxEnergy(fRatioDataSmoothness*LBPInference::MaxEnergy);

				// 设置标签0的成本（未定义）  初始化 也就是为没有对应faceData数据的的进行设置
				FOREACH(s, inferences) {
					LBPInference& inference = inferences[s];
					if (inference.GetNumNodes() == 0)
						continue;
					const NodeID numNodes(sizes[s]);
					for (NodeID nodeID=0; nodeID<numNodes; ++nodeID)
						inference.SetDataCost((Label)0, nodeID, MaxEnergy); // 标签为0的
				}

				// 为所有标签设置数据成本（标签0除外-未定义）
				FOREACH(f, facesDatas) {
					LBPInference& inference = inferences[components[f]];
					if (inference.GetNumNodes() == 0)
						continue;
					const FaceDataArr& faceDatas = facesDatas[f];
					const NodeID nodeID(nodeIDs[f]);

					FOREACHPTR(pFaceData, faceDatas) {
						const FaceData& faceData = *pFaceData;
						const Label label((Label)faceData.idxView+1);
						const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);  // 梯度赋值 / 梯度幅值均值

						const float dataCost((1.f-normalizedQuality)*MaxEnergy);

						inference.SetDataCost(label, nodeID, dataCost); // 设置数据成本
					}
				}
			}

			// 为每个面分配最佳视图（标签）（标签0保留为未定义）  令能量函数最小
			FOREACH(s, inferences) {
				LBPInference& inference = inferences[s];
				if (inference.GetNumNodes() == 0)
					continue;
				inference.Optimize();
			}

			// 提取结果标签 即所对应的视图
			labels.Memset(0xFF);
			FOREACH(l, labels) {
				LBPInference& inference = inferences[components[l]];
				if (inference.GetNumNodes() == 0)
					continue;
				const Label label(inference.GetLabel(nodeIDs[l]));
				ASSERT(label < images.GetSize()+1);
				if (label > 0)
					labels[l] = label-1;
			}
			#endif

			#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
			// initialize inference structures
			const LabelID numLabels(images.GetSize()+1);
			CLISTDEFIDX(TRWSInference, FIndex) inferences(nComponents);
			FOREACH(s, sizes) {
				const NodeID numNodes(sizes[s]);
				ASSERT(numNodes > 0);
				if (numNodes <= 1)
					continue;
				TRWSInference& inference = inferences[s];
				inference.Init(numNodes, numLabels);
			}

			// set data costs
			{
				// add nodes
				CLISTDEF0(EnergyType) D(numLabels);
				FOREACH(f, facesDatas) {
					TRWSInference& inference = inferences[components[f]];
					if (inference.IsEmpty())
						continue;
					D.MemsetValue(MaxEnergy);
					const FaceDataArr& faceDatas = facesDatas[f];
					FOREACHPTR(pFaceData, faceDatas) {
						const FaceData& faceData = *pFaceData;
						const Label label((Label)faceData.idxView);
						const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
						const EnergyType dataCost(MaxEnergy*(1.f-normalizedQuality));
						D[label] = dataCost;
					}
					const NodeID nodeID(nodeIDs[f]);
					inference.AddNode(nodeID, D.Begin());
				}
				// add edges
				EdgeOutIter ei, eie;
				FOREACH(f, faces) {
					TRWSInference& inference = inferences[components[f]];
					if (inference.IsEmpty())
						continue;
					for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
						ASSERT(f == (FIndex)ei->m_source);
						const FIndex fAdj((FIndex)ei->m_target);
						ASSERT(components[f] == components[fAdj]);
						if (f < fAdj) // add edges only once
							inference.AddEdge(nodeIDs[f], nodeIDs[fAdj]);
					}
				}
			}

			// assign the optimal view (label) to each face
			#ifdef TEXOPT_USE_OPENMP
			#pragma omp parallel for schedule(dynamic)
			for (int i=0; i<(int)inferences.GetSize(); ++i) {
			#else
			FOREACH(i, inferences) {
			#endif
				TRWSInference& inference = inferences[i];
				if (inference.IsEmpty())
					continue;
				inference.Optimize();
			}
			// extract resulting labeling
			labels.Memset(0xFF);
			FOREACH(l, labels) {
				TRWSInference& inference = inferences[components[l]];
				if (inference.IsEmpty())
					continue;
				const Label label(inference.GetLabel(nodeIDs[l]));
				ASSERT(label >= 0 && label < numLabels);
				if (label < images.GetSize())
					labels[l] = label;
			}
			#endif
		}

		// 创建纹理贴图
		{
			// 具有相同标号的连通面的子图中的除法图
			EdgeIter ei, eie;
			const PairIdxArr::IDX startLabelSeamEdges(seamEdges.GetSize());
			// 标签不一致 存在缝隙
			for (boost::tie(ei, eie) = boost::edges(graph); ei != eie; ++ei) {
				const FIndex fSource((FIndex)ei->m_source);
				const FIndex fTarget((FIndex)ei->m_target);
				ASSERT(components[fSource] == components[fTarget]);
				if (labels[fSource] != labels[fTarget])
					seamEdges.AddConstruct(fSource, fTarget);   // 缝隙
			}
			// 从图中移除标签不一致的边
			for (const PairIdx *pEdge=seamEdges.Begin()+startLabelSeamEdges, *pEdgeEnd=seamEdges.End(); pEdge!=pEdgeEnd; ++pEdge)
				boost::remove_edge(pEdge->i, pEdge->j, graph);

			// 查找连接的组件:纹理贴图
			ASSERT((FIndex)boost::num_vertices(graph) == components.GetSize());
			const FIndex nComponents(boost::connected_components(graph, components.Begin()));   // 重新计算联通图数量

			// 创建纹理贴图； 最后一个纹理贴图包含所有没有纹理的面
			LabelArr sizes(nComponents);    // 存储每个联通图有多少节点
			sizes.Memset(0);
			FOREACH(c, components)
				++sizes[components[c]];

			texturePatches.Resize(nComponents+1);
			texturePatches.Last().label = NO_ID;
			// 遍历网格 将符合条件的网格存储到texturePatches中
			FOREACH(f, faces) {
				const Label label(labels[f]);
				const FIndex c(components[f]);
				TexturePatch& texturePatch = texturePatches[c];
				ASSERT(texturePatch.label == label || texturePatch.faces.IsEmpty());
				if (label == NO_ID) {
					texturePatch.label = NO_ID;
					texturePatches.Last().faces.Insert(f);
				} else {
					if (texturePatch.faces.IsEmpty()) {
						texturePatch.label = label;
						texturePatch.faces.Reserve(sizes[c]);
					}
					texturePatch.faces.Insert(f);   // 插入网格
				}
			}

			// 删除标签无效的所有补丁（最后一个除外），并创建从旧索引到新索引的映射
			mapIdxPatch.Resize(nComponents);
			std::iota(mapIdxPatch.Begin(), mapIdxPatch.End(), 0);   // 用顺序递增的值赋值指定范围内的元素. 依次为 0, 1, 2, 3...

			// 移除无效的
			for (FIndex t = nComponents; t-- > 0; ) {
				if (texturePatches[t].label == NO_ID) {
					texturePatches.RemoveAtMove(t);
					mapIdxPatch.RemoveAtMove(t);
				}
			}

			const unsigned numPatches(texturePatches.GetSize()-1);
			uint32_t idxPatch(0);

			for (IndexArr::IDX i=0; i< mapIdxPatch.GetSize(); ++i) {
				while (i < mapIdxPatch[i])
					mapIdxPatch.InsertAt(i++, numPatches);
				mapIdxPatch[i] = idxPatch++;
			}
			while (mapIdxPatch.GetSize() <= nComponents)
				mapIdxPatch.Insert(numPatches);
		}
	}
	return true;
}


// 创建接缝顶点和边
void MeshTexture::CreateSeamVertices()
{
	// 每个顶点将包含它所分离的贴片列表，除了包含不可见面的贴片；
	// 每个贴片包含从该顶点开始的属于该纹理贴片的边的列表
	// （通常在每个贴片中有成对的边，表示从该顶点开始的分隔两个有效贴片的两条边）
	VIndex vs[2];
	uint32_t vs0[2], vs1[2];
	std::unordered_map<VIndex, uint32_t> mapVertexSeam;
	const unsigned numPatches(texturePatches.GetSize()-1);

	// 遍历缝隙
	FOREACHPTR(pEdge, seamEdges) {
		// 存储边缘，用于以后的接缝优化
		ASSERT(pEdge->i < pEdge->j);    // pEdge->i 对应的网格索引 pEdge->j 对应的网格索引
		const uint32_t idxPatch0(mapIdxPatch[components[pEdge->i]]);	// 联通分量对应的块
		const uint32_t idxPatch1(mapIdxPatch[components[pEdge->j]]);	// 联通分量对应的块
		ASSERT(idxPatch0 != idxPatch1 || idxPatch0 == numPatches);
		if (idxPatch0 == idxPatch1)
			continue;

		seamVertices.ReserveExtra(2);   // 申请空间
		scene.mesh.GetEdgeVertices(pEdge->i, pEdge->j, vs0, vs1);	// 获取边在图片中的顶点
		ASSERT(faces[pEdge->i][vs0[0]] == faces[pEdge->j][vs1[0]]);
		ASSERT(faces[pEdge->i][vs0[1]] == faces[pEdge->j][vs1[1]]);

		vs[0] = faces[pEdge->i][vs0[0]];    // 顶点
		vs[1] = faces[pEdge->i][vs0[1]];

		const auto itSeamVertex0(mapVertexSeam.emplace(std::make_pair(vs[0], seamVertices.GetSize()))); // 第一个顶点
		if (itSeamVertex0.second)
			seamVertices.AddConstruct(vs[0]);
		SeamVertex& seamVertex0 = seamVertices[itSeamVertex0.first->second];

		const auto itSeamVertex1(mapVertexSeam.emplace(std::make_pair(vs[1], seamVertices.GetSize()))); // 第二个顶点
		if (itSeamVertex1.second)
			seamVertices.AddConstruct(vs[1]);
		SeamVertex& seamVertex1 = seamVertices[itSeamVertex1.first->second];

		if (idxPatch0 < numPatches) {
			const TexCoord offset0(texturePatches[idxPatch0].rect.tl());	// top left
			SeamVertex::Patch& patch00 = seamVertex0.GetPatch(idxPatch0);   // 缝隙在第一个块
			SeamVertex::Patch& patch10 = seamVertex1.GetPatch(idxPatch0);   // 缝隙在第二个块

			ASSERT(patch00.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch00.edges.AddConstruct(itSeamVertex1.first->second).idxFace = pEdge->i;
			patch00.proj = faceTexcoords[pEdge->i*3+vs0[0]]+offset0;

			ASSERT(patch10.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch10.edges.AddConstruct(itSeamVertex0.first->second).idxFace = pEdge->i;
			patch10.proj = faceTexcoords[pEdge->i*3+vs0[1]]+offset0;
		}

		if (idxPatch1 < numPatches) {
			const TexCoord offset1(texturePatches[idxPatch1].rect.tl());	// top left
			SeamVertex::Patch& patch01 = seamVertex0.GetPatch(idxPatch1);
			SeamVertex::Patch& patch11 = seamVertex1.GetPatch(idxPatch1);

			ASSERT(patch01.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch01.edges.AddConstruct(itSeamVertex1.first->second).idxFace = pEdge->j;
			patch01.proj = faceTexcoords[pEdge->j*3+vs1[0]]+offset1;

			ASSERT(patch11.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch11.edges.AddConstruct(itSeamVertex0.first->second).idxFace = pEdge->j;
			patch11.proj = faceTexcoords[pEdge->j*3+vs1[1]]+offset1;
		}
	}
	seamEdges.Release();
}

// 论文 颜色查找区域  todo
void MeshTexture::GlobalSeamLeveling()
{
	ASSERT(!seamVertices.IsEmpty());
	const unsigned numPatches(texturePatches.GetSize()-1);

	// 查找每个顶点的补丁块ID
	PatchIndices patchIndices(vertices.GetSize());
	patchIndices.Memset(0);
	FOREACH(f, faces) {
		const uint32_t idxPatch(mapIdxPatch[components[f]]); // 面联通部分对应的块id
		const Face& face = faces[f];
		for (int v=0; v<3; ++v)
			patchIndices[face[v]].idxPatch = idxPatch;	// 该面的3个顶点赋值
	}

	// 标记缝隙顶点
	FOREACH(i, seamVertices) {
		const SeamVertex& seamVertex = seamVertices[i];
		ASSERT(!seamVertex.patches.IsEmpty());

		PatchIndex& patchIndex = patchIndices[seamVertex.idxVertex];
		patchIndex.bIndex = true;
		patchIndex.idxSeamVertex = i;
	}

	// 将解向量X内的行索引分配给每个顶点/贴片
	ASSERT(vertices.GetSize() < static_cast<VIndex>(std::numeric_limits<MatIdx>::max()));
	MatIdx rowsX(0);
	typedef std::unordered_map<uint32_t,MatIdx> VertexPatch2RowMap;
	cList<VertexPatch2RowMap> vertpatch2rows(vertices.GetSize());

	FOREACH(i, vertices) {
		const PatchIndex& patchIndex = patchIndices[i];	// 缝隙顶点对应的块
		VertexPatch2RowMap& vertpatch2row = vertpatch2rows[i];

		if (patchIndex.bIndex) {	// 缝隙
			// 顶点是多个补丁块的一部分  缝隙
			const SeamVertex& seamVertex = seamVertices[patchIndex.idxSeamVertex];
			ASSERT(seamVertex.idxVertex == i);

			FOREACHPTR(pPatch, seamVertex.patches) {
				ASSERT(pPatch->idxPatch != numPatches);

				vertpatch2row[pPatch->idxPatch] = rowsX++;  // 行计数
			}
		} else
		if (patchIndex.idxPatch < numPatches) {
			// 顶点是1个布丁块的一部分
			vertpatch2row[patchIndex.idxPatch] = rowsX++;
		}
	}

	// 填充Tikhonov Gamma矩阵（正则化约束）
	const float lambda(0.1f);
	MatIdx rowsGamma(0);	// 矩阵元素为 +/- 0.1
	Mesh::VertexIdxArr adjVerts;
	CLISTDEF0(MatEntry) rows(0, vertices.GetSize()*4);

	FOREACH(v, vertices) {
		adjVerts.Empty();
		scene.mesh.GetAdjVertices(v, adjVerts);	// 获取顶点v的相邻顶点

		VertexPatchIterator itV(patchIndices[v], seamVertices); // 顶点块迭代器
		while (itV.Next()) {
			const uint32_t idxPatch(itV);
			if (idxPatch == numPatches)
				continue;

			const MatIdx col(vertpatch2rows[v].at(idxPatch));
			FOREACHPTR(pAdjVert, adjVerts) {
				const VIndex vAdj(*pAdjVert);

				if (v >= vAdj)
					continue;

				VertexPatchIterator itVAdj(patchIndices[vAdj], seamVertices);   // 临近顶点迭代器

				while (itVAdj.Next()) {
					const uint32_t idxPatchAdj(itVAdj);

					if (idxPatch == idxPatchAdj) {
						const MatIdx colAdj(vertpatch2rows[vAdj].at(idxPatchAdj));
						rows.AddConstruct(rowsGamma, col, lambda);
						rows.AddConstruct(rowsGamma, colAdj, -lambda);
						++rowsGamma;
					}
				}
			}
		}
	}
    ASSERT(rows.GetSize()/2 < static_cast<IDX>(std::numeric_limits<MatIdx>::max()));
	// cout << rowsGamma << endl;

	SparseMat Gamma(rowsGamma, rowsX);	// 稀疏矩阵  Gamma  稀疏矩阵 元素为 +/- 0.1
	Gamma.setFromTriplets(rows.Begin(), rows.End());
	rows.Empty();

	// 填充线性方程组的矩阵A和向量B的系数   Ax = b
	IndexArr indices;
	Colors vertexColors;
	Colors coeffB;

	FOREACHPTR(pSeamVertex, seamVertices) {
		const SeamVertex& seamVertex = *pSeamVertex;
		if (seamVertex.patches.GetSize() < 2)
			continue;

		seamVertex.SortByPatchIndex(indices);	// sort
		vertexColors.Resize(indices.GetSize());

		FOREACH(i, indices) {	// 按照块索引遍历
			const SeamVertex::Patch& patch0 = seamVertex.patches[indices[i]];
			ASSERT(patch0.idxPatch < numPatches);

			SampleImage sampler(images[texturePatches[patch0.idxPatch].label].image);

			FOREACHPTR(pEdge, patch0.edges) {
				const SeamVertex& seamVertex1 = seamVertices[pEdge->idxSeamVertex];
				const SeamVertex::Patches::IDX idxPatch1(seamVertex1.patches.Find(patch0.idxPatch));
				ASSERT(idxPatch1 != SeamVertex::Patches::NO_INDEX);
				const SeamVertex::Patch& patch1 = seamVertex1.patches[idxPatch1];
				sampler.AddEdge(patch0.proj, patch1.proj);
			}
			vertexColors[i] = sampler.GetColor();
		}

		const VertexPatch2RowMap& vertpatch2row = vertpatch2rows[seamVertex.idxVertex];
		for (IDX i=0; i<indices.GetSize()-1; ++i) {
			const uint32_t idxPatch0(seamVertex.patches[indices[i]].idxPatch);
			const Color& color0 = vertexColors[i];
			const MatIdx col0(vertpatch2row.at(idxPatch0));

			for (IDX j=i+1; j<indices.GetSize(); ++j) {
				const uint32_t idxPatch1(seamVertex.patches[indices[j]].idxPatch);
				const Color& color1 = vertexColors[j];
				const MatIdx col1(vertpatch2row.at(idxPatch1));
				ASSERT(idxPatch0 < idxPatch1);

				const MatIdx rowA((MatIdx)coeffB.GetSize());
				coeffB.Insert(color1 - color0);
				ASSERT(ISFINITE(coeffB.Last()));

				rows.AddConstruct(rowA, col0,  1.f);
				rows.AddConstruct(rowA, col1, -1.f);
			}
		}
	}
	ASSERT(coeffB.GetSize() < static_cast<IDX>(std::numeric_limits<MatIdx>::max()));

	const MatIdx rowsA((MatIdx)coeffB.GetSize());
	SparseMat A(rowsA, rowsX);	//  A  稀疏矩阵 元素为 +/- 1
	A.setFromTriplets(rows.Begin(), rows.End());
	rows.Release();

	SparseMat Lhs(A.transpose() * A + Gamma.transpose() * Gamma);	// 公式3  (A^T*A + Γ^T*Γ)	左边向量
	// CG只使用下三角形，所以剪枝其余部分并压缩矩阵
	Lhs.prune([](const int& row, const int& col, const float&) -> bool {
		return col <= row;
	});

	// 全局求解校正颜色
	Eigen::Matrix<float,Eigen::Dynamic,3,Eigen::RowMajor> colorAdjustments(rowsX, 3);
	{
		// 初始化 CG solver
		Eigen::ConjugateGradient<SparseMat, Eigen::Lower> solver;
		solver.setMaxIterations(1000);
		// 公差对应于相对残余误差:  |Ax-b|/|b|
		solver.setTolerance(0.0001f);
		solver.compute(Lhs);
		ASSERT(solver.info() == Eigen::Success);

		#ifdef TEXOPT_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int channel=0; channel<3; ++channel) {
			//初始化右边向量
			const Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> > b(coeffB.Begin()->ptr()+channel, rowsA);
			const Eigen::VectorXf Rhs(SparseMat(A.transpose()) * b);	// rou
			// 求解 x
			const Eigen::VectorXf x(solver.solve(Rhs));
			ASSERT(solver.info() == Eigen::Success);
			// 减去均值，因为系统是欠约束的，我们需要最小调整量的解
			Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> >(colorAdjustments.data()+channel, rowsX) = x.array() - x.mean();

			DEBUG_LEVEL(3, "\tcolor channel %d: %d iterations, %g residual", channel, solver.iterations(), solver.error());
		}
	}

	// 使用校正颜色调整纹理贴图
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {
	#endif
		const uint32_t idxPatch((uint32_t)i);

		TexturePatch& texturePatch = texturePatches[idxPatch];
		ColorMap imageAdj(texturePatch.rect.size());
		imageAdj.memset(0);

		// 在整个补丁上插值颜色调整
		RasterPatchColorData data(imageAdj);
		FOREACHPTR(pIdxFace, texturePatch.faces) {
			const FIndex idxFace(*pIdxFace);
			const Face& face = faces[idxFace];

			data.tri = faceTexcoords.Begin()+idxFace*3; // 纹理坐标
			for (int v=0; v<3; ++v)
				data.colors[v] = colorAdjustments.row(vertpatch2rows[face[v]].at(idxPatch));
			// 渲染三角形，并且对于每个像素，使用重心坐标从三角形角内插颜色调整
			ColorMap::RasterizeTriangle(data.tri[0], data.tri[1], data.tri[2], data);
		}
		// 以一个像素宽度扩张，以确保贴片边框稍微平滑
		imageAdj.DilateMean<1>(imageAdj, Color::ZERO);
		// 对补丁图像应用颜色校正
		cv::Mat image(images[texturePatch.label].image(texturePatch.rect));
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				const Color& a = imageAdj(r,c);

				if (a == Color::ZERO)
					continue;

				Pixel8U& v = image.at<Pixel8U>(r,c);
				const Color col(RGB2YCBCR(Color(v)));
				const Color acol(YCBCR2RGB(Color(col+a)));
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(acol[p]), 0, 255);
			}
		}
	}
}

// 设置为1，以便在边框对角线上也扩张（通常不需要）
#define DILATE_EXTRA 0
void MeshTexture::ProcessMask(Image8U& mask, int stripWidth)
{
	typedef Image8U::Type Type;

	// dilate and erode around the border,
	// in order to fill all gaps and remove outside pixels
	// (due to imperfect overlay of the raster line border and raster faces)
	#define DILATEDIR(rd,cd) { \
		Type& vi = mask(r+(rd),c+(cd)); \
		if (vi != border) \
			vi = interior; \
	}
	const int HalfSize(1);
	const int RowsEnd(mask.rows-HalfSize);
	const int ColsEnd(mask.cols-HalfSize);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(mask(r,c));
			if (v != border)
				continue;
			#if DILATE_EXTRA
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw,cw);
					if (vi != border)
						vi = interior;
				}
			}
			#else
			DILATEDIR(-1, 0);
			DILATEDIR(1, 0);
			DILATEDIR(0, -1);
			DILATEDIR(0, 1);
			#endif
		}
	}
	#undef DILATEDIR
	#define ERODEDIR(rd,cd) { \
		const int rl(r-(rd)), cl(c-(cd)), rr(r+(rd)), cr(c+(cd)); \
		const Type vl(mask.isInside(ImageRef(cl,rl)) ? mask(rl,cl) : uint8_t(empty)); \
		const Type vr(mask.isInside(ImageRef(cr,rr)) ? mask(rr,cr) : uint8_t(empty)); \
		if ((vl == border && vr == empty) || (vr == border && vl == empty)) { \
			v = empty; \
			continue; \
		} \
	}
	#if DILATE_EXTRA
	for (int i=0; i<2; ++i)
	#endif
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			ERODEDIR(0, 1);
			ERODEDIR(1, 0);
			ERODEDIR(1, 1);
			ERODEDIR(-1, 1);
		}
	}
	#undef ERODEDIR

	// mark all interior pixels with empty neighbors as border
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if (mask(r-1,c) == empty ||
				mask(r,c-1) == empty ||
				mask(r+1,c) == empty ||
				mask(r,c+1) == empty)
				v = border;
		}
	}

	#if 0
	// mark all interior pixels with border neighbors on two sides as border
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if ((orgMask(r+1,c+0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r+1,c+0) == border && orgMask(r-0,c-1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r-0,c-1) == border))
				v = border;
		}
	}
	}
	#endif

	// compute the set of valid pixels at the border of the texture patch
	#define ISEMPTY(mask, x,y) (mask(y,x) == empty)
	const int width(mask.width()), height(mask.height());
	typedef std::unordered_set<ImageRef> PixelSet;
	PixelSet borderPixels;
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (ISEMPTY(mask, x,y))
				continue;
			// valid border pixels need no invalid neighbors
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				borderPixels.insert(ImageRef(x,y));
				continue;
			}
			// check the direct neighborhood of all invalid pixels
			for (int j=-1; j<=1; ++j) {
				for (int i=-1; i<=1; ++i) {
					// if the valid pixel has an invalid neighbor...
					const int xn(x+i), yn(y+j);
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						ISEMPTY(mask, xn,yn)) {
						// add the pixel to the set of valid border pixels
						borderPixels.insert(ImageRef(x,y));
						goto CONTINUELOOP;
					}
				}
			}
			CONTINUELOOP:;
		}
	}

	// iteratively erode all border pixels
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	typedef std::vector<ImageRef> PixelVector;
	for (int s=0; s<stripWidth; ++s) {
		PixelVector emptyPixels(borderPixels.begin(), borderPixels.end());
		borderPixels.clear();
		// mark the new empty pixels as empty in the mask
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it)
			orgMask(*it) = empty;
		// find the set of valid pixels at the border of the valid area
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it) {
			for (int j=-1; j<=1; j++) {
				for (int i=-1; i<=1; i++) {
					const int xn(it->x+i), yn(it->y+j);
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						!ISEMPTY(orgMask, xn, yn))
						borderPixels.insert(ImageRef(xn,yn));
				}
			}
		}
	}
	#undef ISEMPTY

	// mark all remaining pixels empty in the mask
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (orgMask(y,x) != empty)
				mask(y,x) = empty;
		}
	}
	}

	// mark all border pixels
	for (PixelSet::const_iterator it=borderPixels.cbegin(); it!=borderPixels.cend(); ++it)
		mask(*it) = border;

	#if 0
	// dilate border
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(orgMask(r, c));
			if (v != border)
				continue;
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw, cw);
					if (vi == empty)
						vi = border;
				}
			}
		}
	}
	}
	#endif
}

inline MeshTexture::Color ColorLaplacian(const Image32F3& img, int i) {
	const int width(img.width());
	return img(i-width) + img(i-1) + img(i+1) + img(i+width) - img(i)*4.f;
}

void MeshTexture::PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias)
{
	ASSERT(src.width() == mask.width() && src.width() == dst.width());
	ASSERT(src.height() == mask.height() && src.height() == dst.height());
	ASSERT(src.channels() == 3 && dst.channels() == 3 && mask.channels() == 1);
	ASSERT(src.type() == CV_32FC3 && dst.type() == CV_32FC3 && mask.type() == CV_8U);

	#ifndef _RELEASE
	// check the mask border has no pixels marked as interior
	for (int x=0; x<mask.cols; ++x)
		ASSERT(mask(0,x) != interior && mask(mask.rows-1,x) != interior);
	for (int y=0; y<mask.rows; ++y)
		ASSERT(mask(y,0) != interior && mask(y,mask.cols-1) != interior);
	#endif

	const int n(dst.area());
	const int width(dst.width());

	TImage<MatIdx> indices(dst.size());
	indices.memset(0xff);
	MatIdx nnz(0);
	for (int i = 0; i < n; ++i)
		if (mask(i) != empty)
			indices(i) = nnz++;

	Colors coeffB(nnz);
	CLISTDEF0(MatEntry) coeffA(0, nnz);
	for (int i = 0; i < n; ++i) {
		switch (mask(i)) {
		case border: {
			const MatIdx idx(indices(i));
			ASSERT(idx != -1);
			coeffA.AddConstruct(idx, idx, 1.f);
			coeffB[idx] = (const Color&)dst(i);
		} break;
		case interior: {
			const MatIdx idxUp(indices(i - width));
			const MatIdx idxLeft(indices(i - 1));
			const MatIdx idxCenter(indices(i));
			const MatIdx idxRight(indices(i + 1));
			const MatIdx idxDown(indices(i + width));
			// all indices should be either border conditions or part of the optimization
			ASSERT(idxUp != -1 && idxLeft != -1 && idxCenter != -1 && idxRight != -1 && idxDown != -1);
			coeffA.AddConstruct(idxCenter, idxUp, 1.f);
			coeffA.AddConstruct(idxCenter, idxLeft, 1.f);
			coeffA.AddConstruct(idxCenter, idxCenter,-4.f);
			coeffA.AddConstruct(idxCenter, idxRight, 1.f);
			coeffA.AddConstruct(idxCenter, idxDown, 1.f);
			// set target coefficient
			coeffB[idxCenter] = (bias == 1.f ?
								 ColorLaplacian(src,i) :
								 ColorLaplacian(src,i)*bias + ColorLaplacian(dst,i)*(1.f-bias));
		} break;
		}
	}

	SparseMat A(nnz, nnz);
	A.setFromTriplets(coeffA.Begin(), coeffA.End());
	coeffA.Release();

	#ifdef TEXOPT_SOLVER_SPARSELU
	// use SparseLU factorization
	// (faster, but not working if EIGEN_DEFAULT_TO_ROW_MAJOR is defined, bug inside Eigen)
	const Eigen::SparseLU< SparseMat, Eigen::COLAMDOrdering<MatIdx> > solver(A);
	#else
	// use BiCGSTAB solver
	const Eigen::BiCGSTAB< SparseMat, Eigen::IncompleteLUT<float> > solver(A);
	#endif
	ASSERT(solver.info() == Eigen::Success);
	for (int channel=0; channel<3; ++channel) {
		const Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> > b(coeffB.Begin()->ptr()+channel, nnz);
		const Eigen::VectorXf x(solver.solve(b));
		ASSERT(solver.info() == Eigen::Success);
		for (int i = 0; i < n; ++i) {
			const MatIdx index(indices(i));
			if (index != -1)
				dst(i)[channel] = x[index];
		}
	}
}

// local adjustment with Poisson editing 参考论文16  “Poisson image editing.”
void MeshTexture::LocalSeamLeveling()
{
	ASSERT(!seamVertices.IsEmpty());
	const unsigned numPatches(texturePatches.GetSize()-1);

	// 局部调整纹理贴片，使边界在贴片内平滑延续
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {
	#endif
		const uint32_t idxPatch((uint32_t)i);
		TexturePatch& texturePatch = texturePatches[idxPatch];

		// 提取图像   label 视图索引
		const Image8U3& image0(images[texturePatch.label].image);
		Image32F3 image, imageOrg;
		image0(texturePatch.rect).convertTo(image, CV_32FC3, 1.0/255.0);
		image.copyTo(imageOrg);

		// 补丁块覆盖
		Image8U mask(texturePatch.rect.size());
		{
			mask.memset(0);
			RasterPatchCoverageData data(mask);
			FOREACHPTR(pIdxFace, texturePatch.faces) {
				const FIndex idxFace(*pIdxFace);
				data.tri = faceTexcoords.Begin()+idxFace*3;
				ColorMap::RasterizeTriangle(data.tri[0], data.tri[1], data.tri[2], data);
			}
		}

		// 补丁块边界 -> 邻居补丁块
		const TexCoord offset(texturePatch.rect.tl());  // top left
		// 遍历缝隙顶点  使用缝隙所在的两条边上的颜色均值
	    FOREACHPTR(pSeamVertex, seamVertices) {
			const SeamVertex& seamVertex0 = *pSeamVertex;
			if (seamVertex0.patches.GetSize() < 2)
				continue;

			const uint32_t idxVertPatch0(seamVertex0.patches.Find(idxPatch));   // 当前顶点块索引
			if (idxVertPatch0 == SeamVertex::Patches::NO_INDEX)
				continue;

			const SeamVertex::Patch& patch0 = seamVertex0.patches[idxVertPatch0];   // 当前块
			const TexCoord p0(patch0.proj-offset);

			// 对于属于此补丁的顶点的每条边...
			FOREACHPTR(pEdge0, patch0.edges) {
				// 选择从相邻顶点离开的相同边
				const SeamVertex& seamVertex1 = seamVertices[pEdge0->idxSeamVertex];        // patch0的边 的缝隙顶点
				const uint32_t idxVertPatch0Adj(seamVertex1.patches.Find(idxPatch));        // 查找缝隙顶点所在的块 (相邻顶点块)
				ASSERT(idxVertPatch0Adj != SeamVertex::Patches::NO_INDEX);

				const SeamVertex::Patch& patch0Adj = seamVertex1.patches[idxVertPatch0Adj];
				const TexCoord p0Adj(patch0Adj.proj-offset);

				// 查找共享同一条边（具有相同相邻顶点的边）的另一个贴片
				FOREACH(idxVertPatch1, seamVertex0.patches) {
					if (idxVertPatch1 == idxVertPatch0)
						continue;

					const SeamVertex::Patch& patch1 = seamVertex0.patches[idxVertPatch1];   // 临近块
					const uint32_t idxEdge1(patch1.edges.Find(pEdge0->idxSeamVertex));  // 临近块缝隙顶点所在边
					if (idxEdge1 == SeamVertex::Patch::Edges::NO_INDEX)
						continue;

					const TexCoord& p1(patch1.proj);  // 此顶点在此补丁中的投影

					// 选择属于从相邻顶点离开的第二贴片的相同边
					const uint32_t idxVertPatch1Adj(seamVertex1.patches.Find(patch1.idxPatch));
					ASSERT(idxVertPatch1Adj != SeamVertex::Patches::NO_INDEX);
					const SeamVertex::Patch& patch1Adj = seamVertex1.patches[idxVertPatch1Adj];
					const TexCoord& p1Adj(patch1Adj.proj);

					// 这是分隔两个（有效）补丁的边；
                    // 在此补丁上绘制它作为两个补丁的平均颜色
					const Image8U3& image1(images[texturePatches[patch1.idxPatch].label].image);
					RasterPatchMeanEdgeData data(image, mask, imageOrg, image1, p0, p0Adj, p1, p1Adj);
					Image32F3::DrawLine(p0, p0Adj, data);
					break;
				}
			}
		}

		// 在贴片边界处渲染顶点，与相邻贴片相遇
		const Sampler sampler;
		FOREACHPTR(pSeamVertex, seamVertices) {
			const SeamVertex& seamVertex = *pSeamVertex;
			if (seamVertex.patches.GetSize() < 2)
				continue;
			const uint32_t idxVertPatch(seamVertex.patches.Find(idxPatch));
			if (idxVertPatch == SeamVertex::Patches::NO_INDEX)
				continue;

			AccumColor accumColor;	// 对任意类型进行操作的加权累加器类
			// for each patch...
			FOREACHPTR(pPatch, seamVertex.patches) {
				const SeamVertex::Patch& patch = *pPatch;
				//将其视图添加到顶点平均颜色
				const Image8U3& img(images[texturePatches[patch.idxPatch].label].image);
				accumColor.Add(img.sample<Sampler,Color>(sampler, patch.proj)/255.f, 1.f);
			}
			const SeamVertex::Patch& thisPatch = seamVertex.patches[idxVertPatch];
			const ImageRef pt(ROUND2INT(thisPatch.proj-offset));
			image(pt) = accumColor.Normalized();
			mask(pt) = border;
		}

		// 确保边框是连续的，并且只保留给定大小的外部
		ProcessMask(mask, 20);		// 论文 Poisson Editing
		// 计算纹理贴片混合
		PoissonBlending(imageOrg, image, mask);	// 泊松混合
		// 对补丁图像应用颜色校正
		cv::Mat imagePatch(images[texturePatch.label].image(texturePatch.rect));
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				if (mask(r,c) == empty)
					continue;
				const Color& a = image(r,c);
				Pixel8U& v = imagePatch.at<Pixel8U>(r,c);
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(a[p]*255.f), 0, 255);
			}
		}
	}
}

// 基于全局接缝平整的均匀纹理贴图生成方法  bGlobalSeamLeveling true
// 基于局部接缝平整的均匀纹理贴图边界生成方法 bLocalSeamLeveling true
// 纹理大小应为该值的倍数（0 - 2的倍数）  nTextureSizeMultiple 0
// 指定在决定将新修补程序放置在何处时使用的启发式方法（0-最佳匹配，3-良好速度，100-最佳速度）  nRectPackingHeuristic 3
// 用于未被任何图像覆盖的面部的颜色 colEmpty 0x00FF7F27 r 255 g 127 b 39
void MeshTexture::GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty)
{
    // 不管图像多大，如果我们已经知道图片最外一圈的像素值(约束条件)，以及其它像素点的散度值，我们就能把这个方程给列出来，构建泊松方程，重建图像。
	// 在相应视图中投射补丁并计算纹理坐标和边界框
	const int border(2);
	faceTexcoords.Resize(faces.GetSize()*3);    // 面纹理坐标	 每个面有3个顶点

	#ifdef TEXOPT_USE_OPENMP
	const unsigned numPatches(texturePatches.GetSize()-1);
	#pragma omp parallel for schedule(dynamic)
	// 遍历纹理块  赋值
	for (int_t idx=0; idx<(int_t)numPatches; ++idx) {
		TexturePatch& texturePatch = texturePatches[(uint32_t)idx];
	#else
	for (TexturePatch *pTexturePatch=texturePatches.Begin(), *pTexturePatchEnd=texturePatches.End()-1; pTexturePatch<pTexturePatchEnd; ++pTexturePatch) {
		TexturePatch& texturePatch = *pTexturePatch;
	#endif
		const Image& imageData = images[texturePatch.label];    // label 存储视图索引
		//投射顶点与计算边界框
		AABB2f aabb(true);	// 边界框类
		// 对于纹理块的所有面 利用投影矩阵 将每个面的3个顶点投影到图像坐标中
		FOREACHPTR(pIdxFace, texturePatch.faces) {
			const FIndex idxFace(*pIdxFace);
			const Face& face = faces[idxFace];
			TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;

			// 计算面投影到纹理视图上的坐标
			for (int i=0; i<3; ++i) {
				texcoords[i] = imageData.camera.ProjectPointP(vertices[face[i]]);
				ASSERT(imageData.image.isInsideWithBorder(texcoords[i], border));
				aabb.InsertFull(texcoords[i]);
			}
		}

		// 计算相对纹理坐标
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMin)));
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMax)));

		texturePatch.rect.x = FLOOR2INT(aabb.ptMin[0])-border;  // 边框x
		texturePatch.rect.y = FLOOR2INT(aabb.ptMin[1])-border;  // 边框y

		texturePatch.rect.width = CEIL2INT(aabb.ptMax[0]-aabb.ptMin[0])+border*2;   // width
		texturePatch.rect.height = CEIL2INT(aabb.ptMax[1]-aabb.ptMin[1])+border*2;  // height
		ASSERT(imageData.image.isInside(texturePatch.rect.tl()));
		ASSERT(imageData.image.isInside(texturePatch.rect.br()));

		const TexCoord offset(texturePatch.rect.tl());	// top left

		// 将面的坐标调小 因为边界的存在
		FOREACHPTR(pIdxFace, texturePatch.faces) {	// 一个纹理快包含一个联通块的多个网格
			const FIndex idxFace(*pIdxFace);
			TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;	// 网格有3个顶点
			for (int v=0; v<3; ++v)
				texcoords[v] -= offset;
		}
	}
	{
		// 初始化最后一个补丁指向一个小的统一颜色补丁
		TexturePatch& texturePatch = texturePatches.Last();
		const int sizePatch(border*2+1);	// 5
		texturePatch.rect = cv::Rect(0,0, sizePatch,sizePatch);

		FOREACHPTR(pIdxFace, texturePatch.faces) {
			const FIndex idxFace(*pIdxFace);
			TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;
			for (int i=0; i<3; ++i)
				texcoords[i] = TexCoord(0.5f, 0.5f);	// 坐标为 1 / 2
		}
	}

	// 进行接缝调平
	if (texturePatches.GetSize() > 2 && (bGlobalSeamLeveling || bLocalSeamLeveling)) {
		// 创建接缝顶点和边  todo
		CreateSeamVertices();

		// 执行全局接缝调平
		if (bGlobalSeamLeveling) {
			TD_TIMER_STARTD();
			GlobalSeamLeveling();
			DEBUG_ULTIMATE("\tglobal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}

		// 进行局部接缝调平
		if (bLocalSeamLeveling) {
			TD_TIMER_STARTD();
			LocalSeamLeveling();
			DEBUG_ULTIMATE("\tlocal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}
	}

	// 合并纹理贴图与重叠矩形   计算重叠
	// 遍历贴图块
	for (unsigned i=0; i<texturePatches.GetSize()-2; ++i) {
		TexturePatch& texturePatchBig = texturePatches[i];

		for (unsigned j=1; j<texturePatches.GetSize()-1; ++j) {
			if (i == j)
				continue;
			TexturePatch& texturePatchSmall = texturePatches[j];

			if (texturePatchBig.label != texturePatchSmall.label)	// label - 对应图片id
				continue;

			// 如果A包含在B中的边框上，则返回True。
			if (!RectsBinPack::IsContainedIn(texturePatchSmall.rect, texturePatchBig.rect))	// 边界框
				continue;

			// 平移纹理坐标
			const TexCoord offset(texturePatchSmall.rect.tl()-texturePatchBig.rect.tl());	// 坐标偏移
			// 存在重合 进行合并
			FOREACHPTR(pIdxFace, texturePatchSmall.faces) {
				const FIndex idxFace(*pIdxFace);
				TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;
				for (int v=0; v<3; ++v)
					texcoords[v] += offset;
			}
			// 加入面列表
			texturePatchBig.faces.JoinRemove(texturePatchSmall.faces);
			//移除小补丁
			texturePatches.RemoveAtMove(j--);
		}
	}

	//创建纹理
	{
		// 排列纹理贴片以适合尽可能小的纹理图像
		RectsBinPack::RectArr rects(texturePatches.GetSize());
		FOREACH(i, texturePatches)
			rects[i] = texturePatches[i].rect;
		int textureSize(RectsBinPack::ComputeTextureSize(rects, nTextureSizeMultiple));	//纹理大小

		// 增加纹理大小，直到所有贴片都适合
		while (true) {
			TD_TIMER_STARTD();
			bool bPacked(false);
			const unsigned typeRectsBinPack(nRectPackingHeuristic/100);	// 3 / 100
			const unsigned typeSplit((nRectPackingHeuristic-typeRectsBinPack*100)/10);
			const unsigned typeHeuristic(nRectPackingHeuristic%10);
			switch (typeRectsBinPack) {
			case 0: {
				MaxRectsBinPack pack(textureSize, textureSize);
				bPacked = pack.Insert(rects, (MaxRectsBinPack::FreeRectChoiceHeuristic)typeHeuristic);
				break; }
			case 1: {
				SkylineBinPack pack(textureSize, textureSize, typeSplit!=0);
				bPacked = pack.Insert(rects, (SkylineBinPack::LevelChoiceHeuristic)typeHeuristic);
				break; }
			case 2: {
				GuillotineBinPack pack(textureSize, textureSize);
				bPacked = pack.Insert(rects, false, (GuillotineBinPack::FreeRectChoiceHeuristic)typeHeuristic, (GuillotineBinPack::GuillotineSplitHeuristic)typeSplit);
				break; }
			default:
				ABORT("error: unknown RectsBinPack type");
			}
			DEBUG_ULTIMATE("\tpacking texture completed: %u patches, %u texture-size (%s)", rects.GetSize(), textureSize, TD_TIMER_GET_FMT().c_str());
			if (bPacked)
				break;
			textureSize *= 2;
		}

		// 创建纹理图像
		const float invNorm(1.f/(float)(textureSize-1));
		textureDiffuse.create(textureSize, textureSize);
		textureDiffuse.setTo(cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));

		#ifdef TEXOPT_USE_OPENMP
		#pragma omp parallel for schedule(dynamic)
		for (int_t i=0; i<(int_t)texturePatches.GetSize(); ++i) {
		#else
		FOREACH(i, texturePatches) {
		#endif
			const uint32_t idxPatch((uint32_t)i);	// 块id
			const TexturePatch& texturePatch = texturePatches[idxPatch];	// 纹理块
			const RectsBinPack::Rect& rect = rects[idxPatch];	// 矩形
			// 复制补丁块图像
			ASSERT((rect.width == texturePatch.rect.width && rect.height == texturePatch.rect.height) ||
				   (rect.height == texturePatch.rect.width && rect.width == texturePatch.rect.height));

			int x(0), y(1);
			if (texturePatch.label != NO_ID) {
				const Image& imageData = images[texturePatch.label];

				cv::Mat patch(imageData.image(texturePatch.rect));
				if (rect.width != texturePatch.rect.width) {
					// 翻转贴片和纹理坐标
					patch = patch.t();
					x = 1; y = 0;
				}
				patch.copyTo(textureDiffuse(rect));
			}

			// 计算最终纹理坐标
			const TexCoord offset(rect.tl());
			FOREACHPTR(pIdxFace, texturePatch.faces) {
				const FIndex idxFace(*pIdxFace);
				TexCoord* texcoords = faceTexcoords.Begin()+idxFace*3;
				for (int v=0; v<3; ++v) {
					TexCoord& texcoord = texcoords[v];
					//平移、归一化和翻转Y轴
					texcoord = TexCoord(
						(texcoord[x]+offset.x)*invNorm,
						1.f-(texcoord[y]+offset.y)*invNorm
					);
				}
			}
		}
	}
}

// 纹理网格
bool Scene::TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, float fOutlierThreshold, float fRatioDataSmoothness, bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty)
{
	MeshTexture texture(*this, nResolutionLevel, nMinResolution);

	// 为每个面分配最佳视图
	{
		TD_TIMER_STARTD();
		// 用于查找和删除离群值面部纹理的阈值（0-已禁用）-fOutlierThreshold(6e-2f)
		// 用于调整更紧凑贴片偏好的比率（1-最佳质量/最差紧凑性，~0-最差质量/最佳紧凑性） - fRatioDataSmoothness(0.1f)
		if (!texture.FaceViewSelection(fOutlierThreshold, fRatioDataSmoothness))
			return false;
		DEBUG_EXTRA("Assigning the best view to each face completed: %u faces (%s)", mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// 颜色调整
	{
		TD_TIMER_STARTD();
		// 基于全局接缝平整的均匀纹理贴图生成方法  bGlobalSeamLeveling true
		// 基于局部接缝平整的均匀纹理贴图边界生成方法 bLocalSeamLeveling true
		// 纹理大小应为该值的倍数（0 - 2的倍数）  nTextureSizeMultiple 0
		// 指定在决定将新修补程序放置在何处时使用的启发式方法（0-最佳匹配，3-良好速度，100-最佳速度）  nRectPackingHeuristic 3
		// 用于未被任何图像覆盖的面部的颜色 colEmpty 0x00FF7F27 r 255 g 127 b 39
		texture.GenerateTexture(bGlobalSeamLeveling, bLocalSeamLeveling, nTextureSizeMultiple, nRectPackingHeuristic, colEmpty);
		DEBUG_EXTRA("Generating texture atlas and image completed: %u patches, %u image size (%s)", texture.texturePatches.GetSize(), mesh.textureDiffuse.width(), TD_TIMER_GET_FMT().c_str());
	}

	return true;
} // TextureMesh
/*----------------------------------------------------------------*/
