#define _GLIBCXX_USE_CXX11_ABI 0
#pragma once
#include <string>
#include <vector>
#include <map>

#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#include "../../libs/Common/Types.h"

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/geometry/geometries/register/ring.hpp>

struct MyPoint {
    double x, y/*, z*/;
    MyPoint(double xt, double yt) { x = xt, y = yt; }
    MyPoint() {}
};

BOOST_GEOMETRY_REGISTER_POINT_2D(MyPoint, double, boost::geometry::cs::cartesian, x, y/*, z*/)

typedef boost::geometry::model::polygon<MyPoint> Polygon_2d;
BOOST_GEOMETRY_REGISTER_RING(Polygon_2d::ring_type)

class SceneDevide
{
public:
    SceneDevide(MVS::Scene *sceneOri);
    ~SceneDevide();


    bool InitialParams();
    bool SaveDevidedPointCould();
    bool SceneDevideProcess();
    bool ImageProcess();
    bool PointsCouldProcess();
    bool ImageCrop(const std::vector<Point2d>& range,
                   const std::string & imagePath,
                   const double & averageHeight,
                   std::map<int, int>& matcher,
                   MVS::Scene &scene);
    bool PointCloudCrop(const std::vector<Point2d>& range,
                        std::map<int, int>& matcher,
                        MVS::Scene &scene);
    static bool UniqueImageCamera(MVS::Scene &scene);

    typedef TPoint2<double> Point2d;
    std::vector<MVS::Scene> scenes;
    Point2d boundaryMinXY;
    Point2d boundaryMaxXY;
    unsigned int numOfScenes;
    unsigned int numOfScenesInX;
    unsigned int numOfScenesInY;
    int scaleParam;
    double averageHeight;
    double bufferRange;	// in percentage
    double sceneSizeX;
    double sceneSizeY;
    std::vector<std::vector<Point2d>> sceneRange;
    std::vector<std::map<int, int>> imageIndexMatcher;
    std::string workPath;
    const MVS::Scene *_pScene;

    static int imageWidth;
    static int imageHeight;
};
