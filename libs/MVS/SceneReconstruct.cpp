/*
* SceneReconstruct.cpp
*/

#include "Common.h"
#include "Scene.h"
#include <iostream>
// Delaunay: mesh reconstruction
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Polyhedron_3.h>

using namespace MVS;
using namespace std;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DELAUNAY_USE_OPENMP
#endif

// uncomment to enable reconstruction algorithm of weakly supported surfaces
#define DELAUNAY_WEAKSURF

// uncomment to use IBFS algorithm for max-flow
// (faster, but not clear license policy)
#define DELAUNAY_MAXFLOW_IBFS


// S T R U C T S ///////////////////////////////////////////////////

#ifdef DELAUNAY_MAXFLOW_IBFS
#include "../Math/IBFS/IBFS.h"
template <typename NType, typename VType>
class MaxFlow
{
public:
	// Type-Definitions
	typedef NType node_type;
	typedef VType value_type;
	typedef IBFS::IBFSGraph graph_type;

public:
	MaxFlow(size_t numNodes) {
		graph.initSize((int)numNodes, (int)numNodes*2);
	}

	inline void AddNode(node_type n, value_type source, value_type sink) {
		ASSERT(ISFINITE(source) && source >= 0 && ISFINITE(sink) && sink >= 0);
		graph.addNode((int)n, source, sink);
	}

	inline void AddEdge(node_type n1, node_type n2, value_type capacity, value_type reverseCapacity) {
		ASSERT(ISFINITE(capacity) && capacity >= 0 && ISFINITE(reverseCapacity) && reverseCapacity >= 0);
		graph.addEdge((int)n1, (int)n2, capacity, reverseCapacity);
	}

	value_type ComputeMaxFlow() {
		graph.initGraph();
		return graph.computeMaxFlow();
	}

	inline bool IsNodeOnSrcSide(node_type n) const {
		return graph.isNodeOnSrcSide((int)n);
	}

protected:
	graph_type graph;
};
#else
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
template <typename NType, typename VType>
class MaxFlow
{
public:
	// Type-Definitions
	typedef NType node_type;
	typedef VType value_type;
	typedef boost::vecS out_edge_list_t;
	typedef boost::vecS vertex_list_t;
	typedef boost::adjacency_list_traits<out_edge_list_t, vertex_list_t, boost::directedS> graph_traits;
	typedef typename graph_traits::edge_descriptor edge_descriptor;
	typedef typename graph_traits::vertex_descriptor vertex_descriptor;
	typedef typename graph_traits::vertices_size_type vertex_size_type;
	struct Edge {
		value_type capacity;
		value_type residual;
		edge_descriptor reverse;
	};
	typedef boost::adjacency_list<out_edge_list_t, vertex_list_t, boost::directedS, size_t, Edge> graph_type;
	typedef typename boost::graph_traits<graph_type>::edge_iterator edge_iterator;
	typedef typename boost::graph_traits<graph_type>::out_edge_iterator out_edge_iterator;

public:
	MaxFlow(size_t numNodes) : graph(numNodes+2), S(node_type(numNodes)), T(node_type(numNodes+1)) {}

	void AddNode(node_type n, value_type source, value_type sink) {
		ASSERT(ISFINITE(source) && source >= 0 && ISFINITE(sink) && sink >= 0);
		if (source > 0) {
			edge_descriptor e(boost::add_edge(S, n, graph).first);
			edge_descriptor er(boost::add_edge(n, S, graph).first);
			graph[e].capacity = source;
			graph[e].reverse = er;
			graph[er].reverse = e;
		}
		if (sink > 0) {
			edge_descriptor e(boost::add_edge(n, T, graph).first);
			edge_descriptor er(boost::add_edge(T, n, graph).first);
			graph[e].capacity = sink;
			graph[e].reverse = er;
			graph[er].reverse = e;
		}
	}

	void AddEdge(node_type n1, node_type n2, value_type capacity, value_type reverseCapacity) {
		ASSERT(ISFINITE(capacity) && capacity >= 0 && ISFINITE(reverseCapacity) && reverseCapacity >= 0);
		edge_descriptor e(boost::add_edge(n1, n2, graph).first);
		edge_descriptor er(boost::add_edge(n2, n1, graph).first);
		graph[e].capacity = capacity;
		graph[er].capacity = reverseCapacity;
		graph[e].reverse = er;
		graph[er].reverse = e;
	}

	value_type ComputeMaxFlow() {
		vertex_size_type n_verts(boost::num_vertices(graph));
		color.resize(n_verts);
		std::vector<edge_descriptor> pred(n_verts);
		std::vector<vertex_size_type> dist(n_verts);
		return boost::boykov_kolmogorov_max_flow(graph,
			boost::get(&Edge::capacity, graph),
			boost::get(&Edge::residual, graph),
			boost::get(&Edge::reverse, graph),
			&pred[0],
			&color[0],
			&dist[0],
			boost::get(boost::vertex_index, graph),
			S, T
		);
	}

	inline bool IsNodeOnSrcSide(node_type n) const {
		return (color[n] != boost::white_color);
	}

protected:
	graph_type graph;
	std::vector<boost::default_color_type> color;
	const node_type S;
	const node_type T;
};
#endif
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////

// construct the mesh out of the dense point cloud using Delaunay tetrahedralization & graph-cut method
// see "Exploiting Visibility Information in Surface Reconstruction to Preserve Weakly Supported Surfaces", Jancosek and Pajdla, 2015  02
// 用Delaunay四面体化&图割法从稠密点云中构造网格
// 参见“利用表面重建中的可见性信息来保护弱支撑表面”，Jancosek和Pajdla，2015年。
namespace DELAUNAY {
typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::Point_3 point_t;
typedef kernel_t::Vector_3 vector_t;
typedef kernel_t::Direction_3 direction_t;
typedef kernel_t::Segment_3 segment_t;
typedef kernel_t::Plane_3 plane_t;
typedef kernel_t::Triangle_3 triangle_t;
typedef kernel_t::Ray_3 ray_t;

typedef uint32_t vert_size_t;
typedef uint32_t cell_size_t;

typedef float edge_cap_t;

#ifdef DELAUNAY_WEAKSURF
struct view_info_t;
#endif
struct vert_info_t {
	typedef edge_cap_t Type;
	struct view_t {
		PointCloud::View idxView; // view index
		Type weight; // point's weight
		inline view_t() {}
		inline view_t(PointCloud::View _idxView, Type _weight) : idxView(_idxView), weight(_weight) {}
		inline bool operator <(const view_t& v) const { return idxView < v.idxView; }
		inline operator PointCloud::View() const { return idxView; }
	};
	typedef SEACAVE::cList<view_t,const view_t&,0,4,uint32_t> view_vec_t;
	view_vec_t views; // 单元格向外的面的权重
	#ifdef DELAUNAY_WEAKSURF
	view_info_t* viewsInfo; // 每个视图缓存两个面，从指向摄像机和端部的点开始（仅由支撑薄弱的表面使用）
	inline vert_info_t() : viewsInfo(NULL) {}
	~vert_info_t();
	void AllocateInfo();
	#else
	inline vert_info_t() {}
	#endif
	void InsertViews(const PointCloud& pc, PointCloud::Index idxPoint) {
		const PointCloud::ViewArr& _views = pc.pointViews[idxPoint];
		ASSERT(!_views.IsEmpty());
		const PointCloud::WeightArr* pweights(pc.pointWeights.IsEmpty() ? NULL : pc.pointWeights.Begin()+idxPoint);
		ASSERT(pweights == NULL || _views.GetSize() == pweights->GetSize());
		FOREACH(i, _views) {
			const PointCloud::View viewID(_views[i]);
			const PointCloud::Weight weight(pweights ? (*pweights)[i] : PointCloud::Weight(1));
			// insert viewID in increasing order
			const uint32_t idx(views.FindFirstEqlGreater(viewID));
			if (idx < views.GetSize() && views[idx] == viewID) {
				// the new view is already in the array
				ASSERT(views.FindFirst(viewID) == idx);
				// update point's weight
				views[idx].weight += weight;
			} else {
				// the new view is not in the array,
				// insert it
				views.InsertAt(idx, view_t(viewID, weight));
				ASSERT(views.IsSorted());
			}
		}
	}
};

struct cell_info_t {
	typedef edge_cap_t Type;
	Type f[4]; // 从单元格向外的面的权重
	Type s; // cell's weight towards s-source  朝向s-source的单元格权重
	Type t; // cell's weight towards t-sink  朝向t-sink的单元格权重
	inline const Type* ptr() const { return f; }
	inline Type* ptr() { return f; }
};

typedef CGAL::Triangulation_vertex_base_with_info_3<vert_info_t, kernel_t> vertex_base_t;
typedef CGAL::Triangulation_cell_base_with_info_3<cell_size_t, kernel_t> cell_base_t;
typedef CGAL::Triangulation_data_structure_3<vertex_base_t, cell_base_t> triangulation_data_structure_t;
typedef CGAL::Delaunay_triangulation_3<kernel_t, triangulation_data_structure_t, CGAL::Compact_location> delaunay_t;
typedef delaunay_t::Vertex_handle vertex_handle_t;
typedef delaunay_t::Cell_handle cell_handle_t;
typedef delaunay_t::Facet facet_t;
typedef delaunay_t::Edge edge_t;

#ifdef DELAUNAY_WEAKSURF
struct view_info_t {
	cell_handle_t cell2Cam;
	cell_handle_t cell2End;
};
vert_info_t::~vert_info_t() {
	delete[] viewsInfo;
}
void vert_info_t::AllocateInfo() {
	ASSERT(!views.IsEmpty());
	viewsInfo = new view_info_t[views.GetSize()];
	#ifndef _RELEASE
	memset(viewsInfo, 0, sizeof(view_info_t)*views.GetSize());
	#endif
}
#endif

struct camera_cell_t {
	cell_handle_t cell; // 包含相机的单元格
	std::vector<facet_t> facets; // 摄像机视野内凸壳上的所有面（按重要性排序）
};

struct adjacent_vertex_back_inserter_t {
	const delaunay_t& delaunay;
	const point_t& p;
	vertex_handle_t& v;
	inline adjacent_vertex_back_inserter_t(const delaunay_t& _delaunay, const point_t& _p, vertex_handle_t& _v) : delaunay(_delaunay), p(_p), v(_v) {}
	inline adjacent_vertex_back_inserter_t& operator*() { return *this; }
	inline adjacent_vertex_back_inserter_t& operator++(int) { return *this; }

	inline void operator=(const vertex_handle_t& w) {
		ASSERT(!delaunay.is_infinite(v));

		if (!delaunay.is_infinite(w) && delaunay.geom_traits().compare_distance_3_object()(p, w->point(), v->point()) == CGAL::SMALLER)
			v = w;
	}
};

typedef TPoint3<kernel_t::RT> DPoint3;
template <typename TYPE>
inline TPoint3<TYPE> CGAL2MVS(const point_t& p) {
	return TPoint3<TYPE>((TYPE)p.x(), (TYPE)p.y(), (TYPE)p.z());
}
template <typename TYPE>
inline point_t MVS2CGAL(const TPoint3<TYPE>& p) {
	return point_t((kernel_t::RT)p.x, (kernel_t::RT)p.y, (kernel_t::RT)p.z);
}

// 给定一个面，计算包含它的平面
inline Plane getFacetPlane(const facet_t& facet)    // map pair<Cell_handle, int>
{
	const point_t& v0(facet.first->vertex((facet.second+1)%4)->point());
	const point_t& v1(facet.first->vertex((facet.second+2)%4)->point());
	const point_t& v2(facet.first->vertex((facet.second+3)%4)->point());
	return Plane(CGAL2MVS<REAL>(v0), CGAL2MVS<REAL>(v1), CGAL2MVS<REAL>(v2));
}


// 检查点(P)是否与三角形（A，B，C）共面；
// 返回方向类型
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target ("no-fma")
#endif
static inline int orientation(const point_t& a, const point_t& b, const point_t& c, const point_t& p)
{
	#if 0
	return CGAL::orientation(a, b, c, p);
	#else
	// inexact_orientation
	const double& px = a.x(); const double& py = a.y(); const double& pz = a.z();
	const double pqx(b.x()-px); const double prx(c.x()-px); const double psx(p.x()-px);
	const double pqy(b.y()-py); const double pry(c.y()-py); const double psy(p.y()-py);
	#if 1
	const double det((pqx*pry-prx*pqy)*(p.z()-pz) - (pqx*psy-psx*pqy)*(c.z()-pz) + (prx*psy-psx*pry)*(b.z()-pz));
	const double eps(1e-12);
	#else // very slow due to ABS()
	const double pqz(b.z()-pz); const double prz(c.z()-pz); const double psz(p.z()-pz);
	const double det(CGAL::determinant(
		pqx, pqy, pqz,
		prx, pry, prz,
		psx, psy, psz));
	const double max0(MAXF3(ABS(pqx), ABS(pqy), ABS(pqz)));
	const double max1(MAXF3(ABS(prx), ABS(pry), ABS(prz)));
	const double eps(5.1107127829973299e-15 * MAXF(max0, max1));
	#endif
	if (det >  eps) return CGAL::POSITIVE;
	if (det < -eps) return CGAL::NEGATIVE;
	return CGAL::COPLANAR;
	#endif
}
#ifdef __GNUC__
#pragma GCC pop_options
#endif

//// Check if a point (p) is inside a frustum
//// given the four corners (a, b, c, d) and the origin (o) of the frustum
//inline bool checkPointInside(const point_t& a, const point_t& b, const point_t& c, const point_t& d, const point_t& o, const point_t& p)
//{
//	return (
//		orientation(o, a, b, p) == CGAL::POSITIVE &&
//		orientation(o, b, c, p) == CGAL::POSITIVE &&
//		orientation(o, c, d, p) == CGAL::POSITIVE &&
//		orientation(o, d, a, p) == CGAL::POSITIVE
//	);
//}

// 给定一个单元格和它里面的摄像机，如果单元格是无限的，找到凸壳上和摄像机截头体内的所有面， 否则返回所有四个单元格的面
template <int FacetOrientation>
void fetchCellFacets(const delaunay_t& Tr, const std::vector<facet_t>& hullFacets, const cell_handle_t& cell, const Image& imageData, std::vector<facet_t>& facets)
{
    // 如果单元格是有限的   真，当且仅当c与无穷远点相关。
	if (!Tr.is_infinite(cell)) {
		// 存储单元格的所有4个面
		for (int i=0; i<4; ++i) {
			const facet_t f(cell, i);
			ASSERT(!Tr.is_infinite(f));
			facets.push_back(f);
		}
		return;
	}

	// 在摄像机视图中找到凸壳上的所有面创建4个平面
	ASSERT(facets.empty());
	typedef TFrustum<REAL,4> Frustum;   // 基本截头体类（表示为朝向截头体外部的6个平面）
	Frustum frustum(imageData.camera.P, imageData.width, imageData.width, 0, 1);

	// 遍历所有单元格
	const point_t ptOrigin(MVS2CGAL(imageData.camera.C));

	for (const facet_t& face: hullFacets) {
		// 如果可见，则添加面
		const triangle_t verts(Tr.triangle(face));  // 三角形的方向是使它的法线指向单元格c的内部。
        //检查点(P)是否与三角形（A，B，C）共面；返回方向类型(正 负 共面)
        if (orientation(verts[0], verts[1], verts[2], ptOrigin) != FacetOrientation)
			continue;

		AABB3 ab(CGAL2MVS<REAL>(verts[0]));
		for (int i=1; i<3; ++i)
			ab.Insert(CGAL2MVS<REAL>(verts[i]));

		if (frustum.Classify(ab) == CULLED) // 删除  剔除AABB到N侧截头体，指向外部的法线
			continue;

		facets.push_back(face);
	}
}


// 有关段和面之间的交集的信息
struct intersection_t {
	enum Type {FACET, EDGE, VERTEX};
	cell_handle_t ncell; // 与最后一个相交面相邻的单元格
	vertex_handle_t v1; // vertex for vertex intersection, 1st edge vertex for edge intersection
	vertex_handle_t v2; // 2nd edge vertex for edge intersection
	facet_t facet; // 相交面
	Type type; // type of intersection (inside facet, on edge, or vertex)
	REAL dist; // 从起点（摄像机）到此面的距离
	bool bigger; // 远离起点  or 朝向起点
	const Ray3 ray; // 从起点到终点方向的光线(point->camera/end-point)
	inline intersection_t() {}
	inline intersection_t(const Point3& pt, const Point3& dir) : dist(-FLT_MAX), bigger(true), ray(pt, dir) {}
};

// 检查线段（p，q）是否与三角形（a，b，c）的边共面:
// coplanar[in，out]: 指向与pq共面的边的索引的3个int的数组的指针
// 返回共面中的条目数
inline int checkEdges(const point_t& a, const point_t& b, const point_t& c, const point_t& p, const point_t& q, int coplanar[3])
{
	int nCoplanar(0);
	switch (orientation(p,q,a,b)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 0; // 1
	}
	switch (orientation(p,q,b,c)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 1; // 2
	}
	switch (orientation(p,q,c,a)) {
	case CGAL::POSITIVE: return -1;
	case CGAL::COPLANAR: coplanar[nCoplanar++] = 2; // 3
	}
	return nCoplanar;   // negative 时返回0
}

// 检查面(F)和段(S)之间的交点
//（源自 CGAL::do_intersect in cgal/triangle_3_segment_3_do_intersect.h)
// 共面[out]:指向与(s)共面的边的索引的3int数组的指针
// 如果没有交集 返回-1 或 与线段共面的边数（0=三角形内的交集）
int intersect(const triangle_t& t, const segment_t& s, int coplanar[3])
{
	const point_t& a = t.vertex(0);
	const point_t& b = t.vertex(1);
	const point_t& c = t.vertex(2);
	const point_t& p = s.source();  // 源头 前景
	const point_t& q = s.target();  // 汇点 背景

	switch (orientation(a,b,c,p)) {
	case CGAL::POSITIVE:
		switch (orientation(a,b,c,q)) {
		case CGAL::POSITIVE:
			// 线段位于三角形支撑平面所定义的正开半空间中
			return -1;
		case CGAL::COPLANAR:
			// q属于三角形的支撑平面
            // p以逆时针顺序看到三角形
			return checkEdges(a,b,c,p,q,coplanar);  // 检查线段（p，q）是否与三角形（a，b，c）的边共面
		case CGAL::NEGATIVE:
			// P以逆时针顺序看到三角形
			return checkEdges(a,b,c,p,q,coplanar);  // 检查线段（p，q）是否与三角形（a，b，c）的边共面
		default:
			break;
		}
	case CGAL::NEGATIVE:
		switch (orientation(a,b,c,q)) {
		case CGAL::POSITIVE:
			// q以逆时针的顺序看三角形
			return checkEdges(a,b,c,q,p,coplanar);
		case CGAL::COPLANAR:
			// q属于三角形的支撑平面
            // p以顺时针顺序看到三角形
			return checkEdges(a,b,c,q,p,coplanar);
		case CGAL::NEGATIVE:
			// 该段位于由三角形的支撑平面定义的负开放半空间中
			return -1;
		default:
			break;
		}
	case CGAL::COPLANAR: // P属于三角形的支撑平面
		switch (orientation(a,b,c,q)) {
		case CGAL::POSITIVE:
			// Q以逆时针的顺序看三角形
			return checkEdges(a,b,c,q,p,coplanar);
		case CGAL::COPLANAR:
			// 该段与三角形的支撑平面共面，因为我们知道它在四面体内，它与面相交
            // coplanar[0]=coplanar[1]=coplanar[2]=3；
			return 3;
		case CGAL::NEGATIVE:
			// Q以顺时针的顺序看三角形
			return checkEdges(a,b,c,p,q,coplanar);
		default:
			break;
		}
	}
	ASSERT("should not happen" == NULL);
	return -1;
}

// 查找段(segment)与哪个面相交，并返回要检查的下一个面:
// in_facets[in]:要检查的面的向量
// out_facets[out]:要在下一步检查的面的向量（可以是in_facets)
// out_inter[out]:交集的类型
// 如果找不到交集并且未到达段的末尾，则返回False
bool intersect(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
{
	ASSERT(!in_facets.empty());
	static const int facet_vertex_order[] = {2,1,3,2,2,3,0,2,0,3,1,0,0,1,2,0};  // ???
	int coplanar[3];    // 共面
	const REAL prevDist(inter.dist);    // 点p 向外距离为1
	for (const facet_t& in_facet: in_facets) {
	    // 保证面是有限的
		ASSERT(!Tr.is_infinite(in_facet));  // 即存在点与其对应

		const int nb_coplanar(intersect(Tr.triangle(in_facet), seg, coplanar)); // edge共面数  seg:摄像机到当前点p的距离
//		if(nb_coplanar == 0) {  // 三角形内交集
//            break;
//        }

		if (nb_coplanar >= 0) {
			// 如果交点不在所需方向，则跳过此单元格  线与平面相交求距离
			inter.dist = inter.ray.IntersectsDist(getFacetPlane(in_facet)); // 从原点到无穷远与平面相交
			if ((inter.dist > prevDist) != inter.bigger)
				continue;
			// 面i的顶点 : j = 4 * i, vertices = facet_vertex_order[j,j+1,j+2] 负方向
			inter.facet = in_facet;
			switch (nb_coplanar) {
			case 0: { // 内部交集
				// 面相交
				inter.type = intersection_t::FACET;
				// 现在查找要检查的下一个面，因为相邻单元格中的三个面与原始面不同
				out_facets.clear();
				const cell_handle_t nc(inter.facet.first->neighbor(inter.facet.second));    // 下一个单元格
				ASSERT(!Tr.is_infinite(nc));
				for (int i=0; i<4; ++i)
					if (nc->neighbor(i) != inter.facet.first)
						out_facets.push_back(facet_t(nc, i));
				return true; }
			case 1: {
				// 1条边的共面=相交边
				const int j(4 * inter.facet.second);
				const int i1(j + coplanar[0]);
				inter.type = intersection_t::EDGE;
				inter.v1 = inter.facet.first->vertex(facet_vertex_order[i1+0]);
				//cout << "facet_vertex_order[i1+0]: " << facet_vertex_order[i1+0] << endl;
				inter.v2 = inter.facet.first->vertex(facet_vertex_order[i1+1]);
				//cout << "facet_vertex_order[i1+1]: " << facet_vertex_order[i1+1] << " index: " << i1+1 << endl;
 				// 现在查找要检查的下一个面，作为此单元格中与此边相对的两个面
				out_facets.clear();
				const edge_t out_edge(inter.facet.first, facet_vertex_order[i1+0], facet_vertex_order[i1+1]);
				const typename delaunay_t::Cell_circulator efc(Tr.incident_cells(out_edge));
				typename delaunay_t::Cell_circulator ifc(efc);
				do {
					const cell_handle_t c(ifc);
					if (c == inter.facet.first) continue;
					out_facets.push_back(facet_t(c, c->index(inter.v1)));
					out_facets.push_back(facet_t(c, c->index(inter.v2)));
				} while (++ifc != efc);
				return true; }
			case 2: {
				// 有两条边的共面=碰到一个顶点
                // 查找顶点索引
				const int j(4 * inter.facet.second);
				const int i1(j + coplanar[0]);
				const int i2(j + coplanar[1]);
				int i;
				if (facet_vertex_order[i1] == facet_vertex_order[i2] || facet_vertex_order[i1] == facet_vertex_order[i2+1]) {
					i = facet_vertex_order[i1];
				} else
				if (facet_vertex_order[i1+1] == facet_vertex_order[i2] || facet_vertex_order[i1+1] == facet_vertex_order[i2+1]) {
					i = facet_vertex_order[i1+1];
				} else {
					ASSERT("2 edges intersections without common vertex" == NULL);
				}
				inter.type = intersection_t::VERTEX;
				inter.v1 = inter.facet.first->vertex(i);
				ASSERT(!Tr.is_infinite(inter.v1));
				if (inter.v1->point() == seg.target()) {
					// 已达到目标
					out_facets.clear();
					return false;
				}
				// 现在找到下一个要检查的面，作为与此公共顶点相对的单元格中的面
				out_facets.clear();
				struct cell_back_inserter_t {
					const vertex_handle_t v;
					const cell_handle_t current_cell;
					std::vector<facet_t>& out_facets;
					inline cell_back_inserter_t(const intersection_t& inter, std::vector<facet_t>& _out_facets)
						: v(inter.v1), current_cell(inter.facet.first), out_facets(_out_facets) {}
					inline cell_back_inserter_t& operator*() { return *this; }
					inline cell_back_inserter_t& operator++(int) { return *this; }
					inline void operator=(cell_handle_t c) {
						if (c != current_cell)
							out_facets.push_back(facet_t(c, c->index(v)));
					}
				};
				Tr.finite_incident_cells(inter.v1, cell_back_inserter_t(inter, out_facets));
				return true; }
			}
			// 三边共面=切线=不可能？
			break;
		}
	}
	//
	out_facets.clear();
	return false;
}

//// 与上述相同，但简化为只查找面交点（否则终止）； 如果找到包含段终结点的单元格，或者遇到无限个单元格，则终止
//bool intersectFace(const delaunay_t& Tr, const segment_t& seg, const std::vector<facet_t>& in_facets, std::vector<facet_t>& out_facets, intersection_t& inter)
//{
//	int coplanar[3];
//	for (std::vector<facet_t>::const_iterator it=in_facets.cbegin(); it!=in_facets.cend(); ++it) {
//		ASSERT(!Tr.is_infinite(*it));
//
//		if (intersect(Tr.triangle(*it), seg, coplanar) == 0) {
//			// 面相交
//			inter.facet = *it;
//			inter.type = intersection_t::FACET;
//			// 现在查找要检查的下一个面，因为相邻单元格中的三个面与原始面不同
//			out_facets.clear();
//			inter.ncell = inter.facet.first->neighbor(inter.facet.second);
//
//			// 无限个单元格
//			if (Tr.is_infinite(inter.ncell))
//				return false;
//			for (int i=0; i<4; ++i)
//				if (inter.ncell->neighbor(i) != inter.facet.first)
//					out_facets.push_back(facet_t(inter.ncell, i));
//			return true;
//		}
//	}
//	out_facets.clear();
//	return false;
//}
//// 与上面相同，但从已知的顶点和关联单元开始
//inline bool intersectFace(const delaunay_t& Tr, const segment_t& seg, const vertex_handle_t& v, const cell_handle_t& cell, std::vector<facet_t>& out_facets, intersection_t& inter)
//{
//	if (cell == cell_handle_t())
//		return false;
//	// 如果单元格是无限的
//	if (Tr.is_infinite(cell)) {
//		inter.ncell = inter.facet.first = cell;
//		return true;
//	}
//	std::vector<facet_t>& in_facets = out_facets;
//	ASSERT(in_facets.empty());
//	in_facets.push_back(facet_t(cell, cell->index(v)));
//	return intersectFace(Tr, seg, in_facets, out_facets, inter);
//}


//// 给定一个单元格，计算它的可用空间支持。
//edge_cap_t freeSpaceSupport(const delaunay_t& Tr, const std::vector<cell_info_t>& infoCells, const cell_handle_t& cell)
//{
//	// 将所有4个输入权重相加（对应于相邻单元格的4个面）
//	edge_cap_t wf(0);
//	for (int i=0; i<4; ++i) {
//		const facet_t& mfacet(Tr.mirror_facet(facet_t(cell, i)));
//		wf += infoCells[mfacet.first->info()].f[mfacet.second];
//	}
//	return wf;
//}

// Fetch the triangle formed by the facet vertices,
// making sure the facet orientation is kept (as in CGAL::Triangulation_3::triangle())
// return the vertex handles of the triangle
struct triangle_vhandles_t {
	vertex_handle_t verts[3];
	triangle_vhandles_t() {}
	triangle_vhandles_t(vertex_handle_t _v0, vertex_handle_t _v1, vertex_handle_t _v2)
		#ifdef _SUPPORT_CPP11
		: verts{_v0,_v1,_v2} {}
		#else
		{ verts[0] = _v0; verts[1] = _v1; verts[2] = _v2; }
		#endif
};
inline triangle_vhandles_t getTriangle(cell_handle_t cell, int i)
{
	ASSERT(i >= 0 && i <= 3);
	if ((i&1) == 0)
		return triangle_vhandles_t(
			cell->vertex((i+2)&3),
			cell->vertex((i+1)&3),
			cell->vertex((i+3)&3) );
	return triangle_vhandles_t(
		cell->vertex((i+1)&3),
		cell->vertex((i+2)&3),
		cell->vertex((i+3)&3) );
}

// 计算包含给定刻面的平面与单元的外接球面之间的角度
// return 角的余弦
float computePlaneSphereAngle(const delaunay_t& Tr, const facet_t& facet)
{
	// 计算面法线
	if (Tr.is_infinite(facet.first))
		return 1.f;
	const triangle_vhandles_t tri(getTriangle(facet.first, facet.second));
	const Point3f v0(CGAL2MVS<float>(tri.verts[0]->point()));
	const Point3f v1(CGAL2MVS<float>(tri.verts[1]->point()));
	const Point3f v2(CGAL2MVS<float>(tri.verts[2]->point()));
	const Point3f fn((v1-v0).cross(v2-v0));
		const float fnLenSq(normSq(fn));
		if (fnLenSq == 0.f)
			return 0.5f;

	// 计算其中一个顶点上的外接球面的余切线
	#if CGAL_VERSION_NR < 1041101000
	const Point3f cc(CGAL2MVS<float>(facet.first->circumcenter(Tr.geom_traits())));
	#else
	struct Tools {
		static point_t circumcenter(const delaunay_t& Tr, const facet_t& facet) {
			return Tr.geom_traits().construct_circumcenter_3_object()(
				facet.first->vertex(0)->point(),
				facet.first->vertex(1)->point(),
				facet.first->vertex(2)->point(),
				facet.first->vertex(3)->point()
			);
		}
	};
	const Point3f cc(CGAL2MVS<float>(Tools::circumcenter(Tr, facet)));
	#endif
	const Point3f ct(cc-v0);
	const float ctLenSq(normSq(ct));
	if (ctLenSq == 0.f)
		return 0.5f;

	// 计算两个向量之间的角度
	return CLAMP((fn.dot(ct))/SQRT(fnLenSq*ctLenSq), -1.f, 1.f);
}
} // namespace DELAUNAY


// 首先，如果要插入的点 不比其 至少一个视图中的distinsert像素更接近任何已插入点的投影，则通过逐点插入来迭代创建现有点云的Delaunay三角剖分。
// 接下来，计算由作为顶点的点组成的有向图的所有边的分数。
// 最后，采用 图割算法 对四面体内外表面进行剖分，提取出相应的曲面。
bool Scene::ReconstructMesh(float distInsert, unsigned nItersFixNonManifold,    // 2.5 4
							float kSigma, float kQual, float kb,    // 1 1 4
							float kf, float kRel, float kAbs, float kOutl,  // 3 0.1 1000 400
							float kInf  // 268435456
)
{
	using namespace DELAUNAY;
	ASSERT(!pointcloud.IsEmpty());  // 确定点云不为空
	mesh.Release();

	// 创建Delaunay三角网格
	delaunay_t delaunay;
	std::vector<cell_info_t> infoCells;
	std::vector<camera_cell_t> camCells;
	std::vector<facet_t> hullFacets;
	{
		TD_TIMER_STARTD();

		std::vector<point_t> vertices(pointcloud.points.GetSize()); // 顶点
		std::vector<std::ptrdiff_t> indices(pointcloud.points.GetSize());   // 索引 点云中的第几个
		// 获取点云  初始化顶点及索引
		FOREACH(i, pointcloud.points) {
			const PointCloud::Point& X(pointcloud.points[i]);
			vertices[i] = point_t(X.x, X.y, X.z);
			indices[i] = i;
		}

//        for (int i = 0; i < indices.size(); ++i)
//        {
//            cout << indices[i] << "\t";
//        }
//
//        cout << endl;
		// 顶点排序
		typedef CGAL::Spatial_sort_traits_adapter_3<delaunay_t::Geom_traits, point_t*> Search_traits;
		// 函数spatial_sort()以改进空间局部性的方式对迭代器范围内的点进行排序
		CGAL::spatial_sort(indices.begin(), indices.end(), Search_traits(&vertices[0], delaunay.geom_traits()));
//		for (int i = 0; i < indices.size(); ++i)
//        {
//		    cout << indices[i] << "\t";
//        }
//
//		cout << endl << endl;
		// 从点云中插入顶点  顶点需要满足一定的距离信息
		// 排除条件: 距离 以及 深度相似性
		Util::Progress progress(_T("Points inserted"), indices.size());
		const float distInsertSq(SQUARE(distInsert));	// 2.5f * 2.5f
		vertex_handle_t hint;
		delaunay_t::Locate_type lt;
		int li, lj;

		std::for_each(indices.cbegin(), indices.cend(), [&](size_t idx) {
			const point_t& p = vertices[idx];	// 顶点
			const PointCloud::Point& point = pointcloud.points[idx];	// 顶点
			const PointCloud::ViewArr& views = pointcloud.pointViews[idx];	// 看到该点的图片数组
			ASSERT(!views.IsEmpty());

			if (hint == vertex_handle_t()) {
				// 如果是第一个点，则插入
				hint = delaunay.insert(p);
				ASSERT(hint != vertex_handle_t());	// hint不为空
			} else
			if (distInsert <= 0) {
				// 插入所有的点
				//cout << "here" << endl;
				hint = delaunay.insert(p, hint);
				ASSERT(hint != vertex_handle_t());
			} else {
				// 定位包含此点的单元格
				// If the k-face is a cell, li and lj have no meaning;
				// if it is a facet (resp. vertex), li gives the index of the facet (resp. vertex) and lj has no meaning;
				// if it is and edge, li and lj give the indices of its vertices.
				const cell_handle_t c(delaunay.locate(p, lt, li, lj, hint->cell()));

				if (lt == delaunay_t::VERTEX) { // 0  如果是顶点 location type
					// 重复点，没有要插入的内容，只需更新其可见性信息
					hint = c->vertex(li);
					ASSERT(hint != delaunay.infinite_vertex());
				} else {
					// 找到最近的顶点
					vertex_handle_t nearest;    // 最近的顶点
					if (delaunay.dimension() < 3) {
						// 如果维度小于3，则使用暴力算法
						delaunay_t::Finite_vertices_iterator vit = delaunay.finite_vertices_begin();
						nearest = vit;
						++vit;
						adjacent_vertex_back_inserter_t inserter(delaunay, p, nearest); // 选择在哪个三角网格中进行赋值
						// 单纯比较距离进行更新  vertex_handle_t& v  -- > nearest ;
						for (delaunay_t::Finite_vertices_iterator end = delaunay.finite_vertices_end(); vit != end; ++vit)
							inserter = vit; //　重写操作运算符，如果更小，则赋值
					} else {
                        //-从所定位的单元格的最近顶点开始
                        //-重复获取最近的关联顶点（如果有）
                        //-如果没有，结束
						ASSERT(c != cell_handle_t());
						nearest = delaunay.nearest_vertex_in_cell(p, c);    // 返回单元格c中最接近p的有限顶点
						while (true) {
							const vertex_handle_t v(nearest);
							delaunay.adjacent_vertices(nearest, adjacent_vertex_back_inserter_t(delaunay, p, nearest));
							if (v == nearest)
								break;
						}
					}
					ASSERT(nearest == delaunay.nearest_vertex(p, hint->cell()));
					hint = nearest;
					// 检查点到所有现有点的距离是否足够远
					FOREACHPTR(pViewID, views) {
						const Image& imageData = images[*pViewID];
						const Point3f pn(imageData.camera.ProjectPointP3(point));   // 当前点
						const Point3f pe(imageData.camera.ProjectPointP3(CGAL2MVS<float>(nearest->point())));   // 最邻近点
						// 深度?? || 距离
						if (!IsDepthSimilar(pn.z, pe.z) || normSq(Point2f(pn)-Point2f(pe)) > distInsertSq) {
							//　最近点距离现有点足够远，插入一个新点
							hint = delaunay.insert(p, lt, c, li, lj);   // insert
							ASSERT(hint != vertex_handle_t());
							break;
						}
					}
				}
			}
			// 更新可见性信息
			hint->info().InsertViews(pointcloud, idx);
			++progress;
		});
		progress.close();
		pointcloud.Release();


		// 初始化单元格,对所有单元格进行加权和循环，并存储无限单元格的有限面
		const size_t numNodes(delaunay.number_of_cells());  // 四面体节点数量
		infoCells.resize(numNodes);
		memset(&infoCells[0], 0, sizeof(cell_info_t)*numNodes);
		cell_size_t ciID(0);

		for (delaunay_t::All_cells_iterator ci=delaunay.all_cells_begin(), eci=delaunay.all_cells_end(); ci!=eci; ++ci, ++ciID) {
			ci->info() = ciID;
			// 如果单元格是有限的
			if (!delaunay.is_infinite(ci))
				continue;
			// 在无限单元格中找有限面并存储  4面体
			for (int f=0; f<4; ++f) {
				const facet_t facet(ci, f);
				// 如果是有限的
				if (!delaunay.is_infinite(facet)) {
					// 存储face 与有限顶点关联的face
					hullFacets.push_back(facet);
					break;
				}
			}
		}

		// 查找包含摄像机的所有单元格
		camCells.resize(images.GetSize());
		FOREACH(i, images) {
			const Image& imageData = images[i];
			if (!imageData.IsValid())
				continue;

			const Camera& camera = imageData.camera;
			camera_cell_t& camCell = camCells[i];
			camCell.cell = delaunay.locate(MVS2CGAL(camera.C)); // MVS2CGAL（类型转换）
			ASSERT(camCell.cell != cell_handle_t());

			fetchCellFacets<CGAL::POSITIVE>(delaunay, hullFacets, camCell.cell, imageData, camCell.facets);
			// 将照相机包含的所有单元格链接到源 source
			for (const facet_t& f: camCell.facets)
				infoCells[f.first->info()].s = kInf;    // kInf = 1.0  图切割中质量权重的乘子调整
		}

		DEBUG_EXTRA("Delaunay tetrahedralization completed: %u points -> %u vertices, %u (+%u) cells, %u (+%u) faces (%s)", indices.size(), delaunay.number_of_vertices(), delaunay.number_of_finite_cells(), delaunay.number_of_cells()-delaunay.number_of_finite_cells(), delaunay.number_of_finite_facets(), delaunay.number_of_facets()-delaunay.number_of_finite_facets(), TD_TIMER_GET_FMT().c_str());
	}

	// 对于每个摄像机点射线，将其与四面体相交，并将alpha_vis(point)添加到图中单元格的有向边
    {
        TD_TIMER_STARTD();

        // 估计最小可重构对象的大小
        FloatArr distsSq(0, delaunay.number_of_edges());    // 存储四面体边的长度

        // 边的长度
        for (delaunay_t::Finite_edges_iterator ei = delaunay.finite_edges_begin(), eei = delaunay.finite_edges_end();
             ei != eei; ++ei) {
            const cell_handle_t &c(ei->first);
            // x * x + y * y  像素距离
            distsSq.Insert(normSq(CGAL2MVS<float>(c->vertex(ei->second)->point()) -
                                  CGAL2MVS<float>(c->vertex(ei->third)->point())));
        }

        const float sigma(SQRT(distsSq.GetMedian()) * kSigma);    // sigma常量  等于边长中位数 代表最小可重建对象  论文
        const float inv2SigmaSq(0.5f / (sigma * sigma));    // 1 / (2 * sigma * sigma)
        distsSq.Release();

        std::vector<facet_t> facets;

        // 计算每条边的权重
        {
            TD_TIMER_STARTD();
            Util::Progress progress(_T("Points weighted"), delaunay.number_of_vertices());
#ifdef DELAUNAY_USE_OPENMP
            delaunay_t::Vertex_iterator vertexIter(delaunay.vertices_begin());  // 迭代器  顶点
            const int64_t nVerts(delaunay.number_of_vertices() + 1);
            // 遍历边
#pragma omp parallel for private(facets)
            for (int64_t i = 0; i < nVerts; ++i) {
                delaunay_t::Vertex_iterator vi;
#pragma omp critical
                vi = vertexIter++;
#else
                for (delaunay_t::Vertex_iterator vi=delaunay.vertices_begin(), vie=delaunay.vertices_end(); vi!=vie; ++vi) {
#endif
                vert_info_t &vert(vi->info());  // vert
                if (vert.views.IsEmpty())
                    continue;
#ifdef DELAUNAY_WEAKSURF
                vert.AllocateInfo();    // 申请空间
#endif
                const point_t &p(vi->point());
                const Point3 pt(CGAL2MVS<REAL>(p));

                // 顶点对应的视图
                FOREACH(v, vert.views) {    // 嵌套结构体
                    const typename vert_info_t::view_t view(vert.views[v]);
                    const uint32_t imageID(view.idxView);   // 视图索引
                    const edge_cap_t alpha_vis(view.weight);    // 权重   vert_info_t.weight
                    const Image &imageData = images[imageID];   // scene.images
                    ASSERT(imageData.IsValid());
                    const Camera &camera = imageData.camera;
                    const camera_cell_t &camCell = camCells[imageID];

                    // 计算用于查找点交的光线
                    const Point3 vecCamPoint(pt - camera.C);  // line of sight 视线   ==  向量 == [点 - 相机光心坐标]
                    const REAL invLenCamPoint(REAL(1) / norm(vecCamPoint));   // 1 / 距离(点到相机的距离)
                    intersection_t inter(pt, Point3(vecCamPoint * invLenCamPoint)); // 当前点pt -> 点（距离等于1） 视线

                    // 查找与摄像机点段相交的面
                    const segment_t segCamPoint(MVS2CGAL(camera.C), p); // 摄像机 -> 点p
                    if (!intersect(delaunay, segCamPoint, camCell.facets, facets, inter))   // 相交类型
                        continue;
                    do {
                        // 分配分数，按点到交叉点的距离加权
                        const edge_cap_t w(
                                alpha_vis * (1.f - EXP(-SQUARE((float) inter.dist) * inv2SigmaSq)));  // 论文33 公式2
                        edge_cap_t &f(infoCells[inter.facet.first->info()].f[inter.facet.second]);
#ifdef DELAUNAY_USE_OPENMP
#pragma omp atomic
#endif
                        f += w;
                    } while (intersect(delaunay, segCamPoint, facets, facets, inter));
                    ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
#ifdef DELAUNAY_WEAKSURF
                    ASSERT(vert.viewsInfo[v].cell2Cam == NULL);
                    vert.viewsInfo[v].cell2Cam = inter.facet.first;
#endif


                    // 查找与端点-点段相交的面
                    inter.dist = FLT_MAX;
                    inter.bigger = false;
                    const Point3 endPoint(pt + vecCamPoint * (invLenCamPoint * sigma));
                    const segment_t segEndPoint(MVS2CGAL(endPoint), p); // 端点 -> 点p
                    const cell_handle_t endCell(delaunay.locate(segEndPoint.source(), vi->cell())); // 定位端点所属单元格
                    ASSERT(endCell != cell_handle_t());

                    // 找到单元格的四个面
                    fetchCellFacets<CGAL::NEGATIVE>(delaunay, hullFacets, endCell, imageData, facets);
                    edge_cap_t &t(infoCells[endCell->info()].t);
#ifdef DELAUNAY_USE_OPENMP
#pragma omp atomic
#endif
                    t += alpha_vis;
                    while (intersect(delaunay, segEndPoint, facets, facets, inter)) {
                        // 分配分数，按点到交叉点的距离加权
                        const facet_t &mf(delaunay.mirror_facet(inter.facet));
                        const edge_cap_t w(
                                alpha_vis * (1.f - EXP(-SQUARE((float) inter.dist) * inv2SigmaSq)));    // 论文33 公式2
                        edge_cap_t &f(infoCells[mf.first->info()].f[mf.second]);
#ifdef DELAUNAY_USE_OPENMP
#pragma omp atomic
#endif
                        f += w;
                    }
                    // ASSERT(facets.empty() && inter.type == intersection_t::VERTEX && inter.v1 == vi);
                    ASSERT(facets.empty());
                    // cout << inter.type << endl;
                    // ASSERT(inter.type == intersection_t::VERTEX );
                    // ASSERT(inter.v1 == vi);
#ifdef DELAUNAY_WEAKSURF
                    ASSERT(vert.viewsInfo[v].cell2End == NULL);
                    vert.viewsInfo[v].cell2End = inter.facet.first;
#endif
                }
                ++progress;
            }
            progress.close();
            DEBUG_ULTIMATE("\tweighting completed in %s", TD_TIMER_GET_FMT().c_str());
        }
        camCells.clear();

        // 运行Graph-cut并提取网格
        {
            TD_TIMER_STARTD();

            // 创建图
            MaxFlow<cell_size_t, edge_cap_t> graph(delaunay.number_of_cells());

            // 设置权重
            for (delaunay_t::All_cells_iterator ci = delaunay.all_cells_begin(), ce = delaunay.all_cells_end();
                 ci != ce; ++ci) {
                const cell_size_t ciID(ci->info());
                const cell_info_t &ciInfo(infoCells[ciID]);
                graph.AddNode(ciID, ciInfo.s, ciInfo.t);    // 添加源点 汇点

                for (int i = 0; i < 4; ++i) {
                    const cell_handle_t cj(ci->neighbor(i));
                    const cell_size_t cjID(cj->info());
                    if (cjID < ciID) continue;
                    const cell_info_t &cjInfo(infoCells[cjID]);
                    const int j(cj->index(ci));
                    const edge_cap_t q((1.f - MINF(computePlaneSphereAngle(delaunay, facet_t(ci, i)),
                                                   computePlaneSphereAngle(delaunay, facet_t(cj, j)))) * kQual);
                    graph.AddEdge(ciID, cjID, ciInfo.f[i] + q, cjInfo.f[j] + q);    // 添加边
                }
            }
            infoCells.clear();

            // 查找Graph-cut解决方案  todo
            const float maxflow(graph.ComputeMaxFlow());    // 论文24  2

            // 提取内部/外部单元格之间的小平面形成的表面
            const size_t nEstimatedNumVerts(delaunay.number_of_vertices()); // 顶点个数
            std::unordered_map<void *, Mesh::VIndex> mapVertices;
#if defined(_MSC_VER) && (_MSC_VER > 1600)
            mapVertices.reserve(nEstimatedNumVerts);
#endif
            mesh.vertices.Reserve((Mesh::VIndex) nEstimatedNumVerts);
            mesh.faces.Reserve((Mesh::FIndex) nEstimatedNumVerts * 2); // 面是顶点的2倍 ?

            // 遍历网格  todo
            for (delaunay_t::All_cells_iterator ci = delaunay.all_cells_begin(), ce = delaunay.all_cells_end();
                 ci != ce; ++ci) {
                const cell_size_t ciID(ci->info());
                // 一个网格四面体有四个面
                for (int i = 0; i < 4; ++i) {
                    // 单元格是无限的，则继续
                    if (delaunay.is_infinite(ci, i)) continue;

                    const cell_handle_t cj(ci->neighbor(i));
                    const cell_size_t cjID(cj->info()); // 邻居
                    if (ciID < cjID) continue;

                    const bool ciType(graph.IsNodeOnSrcSide(ciID)); // 当前节点类型 (s or t)
                    if (ciType == graph.IsNodeOnSrcSide(cjID)) continue;    // 邻居节点类型应当与当前节点类型不一样

                    Mesh::Face &face = mesh.faces.AddEmpty();
                    const triangle_vhandles_t tri(getTriangle(ci, i));  // 当前单元格 第i面
                    // 遍历三角形三条边add_tweights
                    for (int v = 0; v < 3; ++v) {
                        const vertex_handle_t vh(tri.verts[v]);
                        ASSERT(vh->point() == delaunay.triangle(ci, i)[v]);

                        const auto pairItID(mapVertices.insert(
                                std::make_pair(vh.for_compact_container(), (Mesh::VIndex) mesh.vertices.GetSize())));

                        if (pairItID.second)
                            mesh.vertices.Insert(CGAL2MVS<Mesh::Vertex::Type>(vh->point()));

                        ASSERT(pairItID.first->second < mesh.vertices.GetSize());
                        face[v] = pairItID.first->second;
                    }
                    // 正确的面定位
                    if (!ciType)
                        std::swap(face[0], face[2]);
                }
            }
            delaunay.clear();

            DEBUG_EXTRA("Delaunay tetrahedras graph-cut completed (%g flow): %u vertices, %u faces (%s)", maxflow,
                        mesh.vertices.GetSize(), mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());
        }

        // 固定非流形顶点和边
        for (unsigned i = 0; i < nItersFixNonManifold; ++i)
            if (!mesh.FixNonManifold()) // Mesh.cpp
                break;
        return true;
    }
}
/*----------------------------------------------------------------*/
