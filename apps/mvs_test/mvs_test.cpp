#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"

#include "SceneDevide.h"

using namespace MVS;

int main(int argc, LPCTSTR* argv)
{
    //test
    //{
    //	Eigen::Matrix<double, 3, 4> pMat;
    //	pMat << 5544.3333513004009, 6327.0118781724641, -4163.8803077455887, -24333702917.6702380000000,
    //		6125.8059861543297, -5628.0998371394608, -3405.3217288966198, 15747638037.5692670000000,
    //		-0.0362355312611, 0.0203282471373, -0.9991365015065, -48380.1370322157820;
    //	Matrix3x4 p(pMat);
    //	Camera cam(p);
    //	cam.DecomposeP();
    //	std::cout << cam.C << std::endl;
    //	getchar();
    //}
    Scene scene;
    // load and estimate a dense point-cloud
    if (!scene.Load("/home/jxq/CODE/output1/mvs/scene_dense.mvs"))
        return EXIT_FAILURE;
    if (scene.pointcloud.IsEmpty()) {
        VERBOSE("error: empty initial point-cloud");
        return EXIT_FAILURE;
    }
    std::cout << scene.images.size() << std::endl;
    SceneDevide::UniqueImageCamera(scene);
    SceneDevide processer(&scene);
    processer.workPath = "/home/jxq/CODE/output/WorkPath303";
    processer.numOfScenesInX = 5;
    processer.numOfScenesInY = 5;
    processer.boundaryMinXY = Point2d(-8.0, -8.0);
    processer.boundaryMaxXY = Point2d(8.0, 8.0);
    processer.InitialParams();
    std::cout << processer.scenes.at(0).images.size() << std::endl;
    //processer.SceneDevideProcess();
    processer.ImageProcess();
    std::cout << processer.scenes.at(0).images.size() << std::endl;
    for (size_t i = 0; i < 4; i++)
    {
        std::cout << processer.sceneRange.at(i).at(0) << " "<< processer.sceneRange.at(i).at(1) << std::endl;
    }
    processer.PointsCouldProcess();
    //for (size_t sceneIndex = 0; sceneIndex < processer.scenes.size(); sceneIndex++)
    //{
    //	processer.PointCloudCrop(processer.sceneRange.at(sceneIndex), processer.imageIndexMatcher.at(sceneIndex), processer.scenes.at(sceneIndex));
    //}
    processer.SaveDevidedPointCould();
    std::cout << processer.scenes.at(0).images.size() << std::endl;

    std::cout << "scene subdevide finished, press to exit" << std::endl;
    getchar();
    return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/