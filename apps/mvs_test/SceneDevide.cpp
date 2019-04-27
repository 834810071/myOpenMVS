#define _GLIBCXX_USE_CXX11_ABI 0

#include "SceneDevide.h"
#include "./stlplus3/filesystemSimplified/file_system.hpp"


using namespace std;
using namespace MVS;

//const int SceneDevide::imageWidth = 8176;
//const int SceneDevide::imageHeight = 6132;
int SceneDevide::imageWidth;
int SceneDevide::imageHeight;
//const int SceneDevide::imageWidth = 1022;
//const int SceneDevide::imageHeight = 766;

bool ShowImageInfo(const MVS::Scene &scene, std::string fileName)
{
    std::ofstream writer(fileName);
    for (auto imageIndexed = scene.images.begin(); imageIndexed != scene.images.end(); imageIndexed++)
    {
        imageIndexed->UpdateCamera(scene.platforms);
        writer << imageIndexed->name << imageIndexed->height << " " << imageIndexed->width << endl
               << imageIndexed->camera.K << endl
               << scene.platforms[imageIndexed->platformID].cameras[imageIndexed->cameraID].K << endl << endl;

        for (auto neightBorIndexed = imageIndexed->neighbors.begin(); neightBorIndexed != imageIndexed->neighbors.end(); neightBorIndexed++)
        {
            writer << neightBorIndexed->idx.ID << " ";
        }
        writer << endl;
    }
    return true;
}

// make all the image have its own camera in platform, desigined only for scene contaions only 1 camera in each platform
bool SceneDevide::UniqueImageCamera(MVS::Scene &scene)
{
    const int numOfPlatform = scene.platforms.size();
    //check wheather the scene contaions only 1 camera in each platform
    for (size_t i = 0; i < numOfPlatform; i++)
    {
        if (scene.platforms[i].cameras.size() != 1)
        {
            std::cout << "Error: failed to unique image camera" << endl;
            return false;
        }
    }
    vector<vector<int>> cameraIndex;
    cameraIndex.resize(numOfPlatform);
    for (size_t imageIndex = 0; imageIndex < scene.images.size(); imageIndex++)
    {
        cameraIndex.at(scene.images[imageIndex].platformID).push_back(imageIndex);
    }
    for (size_t platformIndex = 0; platformIndex < numOfPlatform; platformIndex++)
    {
        Camera cam = scene.platforms[platformIndex].cameras[0];
        for (size_t index = 0; index < cameraIndex.at(platformIndex).size(); index++)
        {
            if (index != 0)
            {
                scene.platforms[platformIndex].cameras.push_back(cam);
            }
            scene.images[cameraIndex.at(platformIndex).at(index)].cameraID = index;
        }
    }
    return true;
}

SceneDevide::SceneDevide(MVS::Scene *sceneOri) :_pScene(sceneOri)
{
}


SceneDevide::~SceneDevide()
{
}

bool SceneDevide::InitialParams()
{
    if (numOfScenesInX == 0 || numOfScenesInY == 0)
    {
        std::cout << "please specific the number of scene in both x and y direcition" << endl;
        return false;
    }
    sceneSizeX = (boundaryMaxXY.x - boundaryMinXY.x) / numOfScenesInX;
    sceneSizeY = (boundaryMaxXY.y - boundaryMinXY.y) / numOfScenesInY;
    numOfScenes = numOfScenesInX*numOfScenesInY;
    scenes.resize(numOfScenes);
    for (size_t sceneIndex = 0; sceneIndex < scenes.size(); sceneIndex++)
    {
        scenes.at(sceneIndex).platforms = (*_pScene).platforms;
        //scenes.at(sceneIndex)
    }
    imageIndexMatcher.resize(numOfScenes);

    bufferRange = 0.1;
    averageHeight = 0.3500;
    //imageWidth = 4088;
    //imageHeight = 3066;
    imageWidth = 8176;
    imageHeight = 6132;

    return true;
}

bool SceneDevide::SaveDevidedPointCould()
{
    for (size_t sceneIndex = 0; sceneIndex < scenes.size(); sceneIndex++)
    {
        string fileName = workPath + "\\pointcould_" + std::to_string(sceneIndex) + ".ply";
        scenes.at(sceneIndex).pointcloud.Save(fileName);
        string sceneName = workPath + "\\block_" + std::to_string(sceneIndex) + ".mvs";
        scenes.at(sceneIndex).Save(sceneName);
    }
    return true;
}

bool SceneDevide::SceneDevideProcess()
{
    //process one of the scene testblock
    {
        //apply the original params to created scene
        MVS::Scene scene;
        scene.platforms = (*_pScene).platforms;
        std::vector<Point2d> range;
        range.push_back(Point2d(-2.6054, -0.9664));
        range.push_back(Point2d(0.5879, 1.0688));
        const double avgHeight(0.3500);
        std::map<int, int> matcher;
        string imagePath("F:\\MillerWorkPath\\VSProject\\openMVS_TEST\\openMVS_TEST\\imageCrop\\imageCrop\\");

        this->ImageCrop(range, imagePath, avgHeight, matcher, scene);
        this->PointCloudCrop(range, matcher, scene);

        std::cout << scene.images.begin()->name << endl;
        std::cout << scene.images.begin()->camera.K;
        scene.Save("test.mvs");
        scene.pointcloud.Save("test.ply");

        //ShowImageInfo(scene, "sceneImageCroped.txt");
        //ShowImageInfo(*_pScene, "sceneImage.txt");
        //for (size_t i = 0; i < _pScene->images[0].neighbors.size(); i++)
        //{
        //	std::cout << _pScene->images[_pScene->images[0].neighbors[i].idx.ID].name << endl;
        //	std::cout << scene.images[scene.images[0].neighbors[i].idx.ID].name << endl;
        //	getchar();
        //}
        //
        std::cout << "process finished" << endl;
        getchar();
    }
    return true;
}

bool SceneDevide::ImageProcess()
{
    for (size_t indexY = 0; indexY < numOfScenesInY; indexY++)
    {
        for (size_t indexX = 0; indexX < numOfScenesInX; indexX++)
        {
            vector<Point2d> range;
            Point2d rangeMin, rangeMax;
            rangeMin.x = boundaryMinXY.x + (indexX - bufferRange)*sceneSizeX;
            rangeMin.y = boundaryMinXY.y + (indexY - bufferRange)*sceneSizeY;
            rangeMax.x = boundaryMinXY.x + (indexX + 1 + bufferRange)*sceneSizeX;
            rangeMax.y = boundaryMinXY.y + (indexY + 1 + bufferRange)*sceneSizeY;
            range.push_back(rangeMin);
            range.push_back(rangeMax);
            sceneRange.push_back(range);
        }
    }
    int sceneIndex(0);
    for (size_t indexY = 0; indexY < numOfScenesInY; indexY++)
    {
        for (size_t indexX = 0; indexX < numOfScenesInX; indexX++)
        {
            sceneIndex = indexY*numOfScenesInX+indexX;
            string imagePath = workPath + "\\block_" + std::to_string(indexY) + std::to_string(indexX);
            ImageCrop(sceneRange.at(sceneIndex), imagePath, averageHeight, imageIndexMatcher.at(sceneIndex), scenes.at(sceneIndex));
        }
    }
    return true;
}

bool SceneDevide::PointsCouldProcess()
{
    if (_pScene->pointcloud.GetSize() == 0)
    {
        std::cout << "Error: Enmpty point cloud in scene" << endl;
        return false;
    }

    if (_pScene->pointcloud.points.size() != _pScene->pointcloud.pointViews.size() ||
        _pScene->pointcloud.points.size() != _pScene->pointcloud.colors.size() ||
        _pScene->pointcloud.points.size() != _pScene->pointcloud.pointWeights.size())
    {
        std::cout << "Error: Invalid point cloud in scene" << endl;
        return false;
    }

    //FIXME: compare which is faster iterater or [] operation
    auto pointView = _pScene->pointcloud.pointViews.begin();
    auto pointColor = _pScene->pointcloud.colors.begin();
    auto pointWeight = _pScene->pointcloud.pointWeights.begin();

    long count(0);
    for (auto point = _pScene->pointcloud.points.begin(); point != _pScene->pointcloud.points.end(); ++point)
    {
        count++;
        int indexX = (point->x - boundaryMinXY.x) / sceneSizeX;
        int indexY = (point->y - boundaryMinXY.y) / sceneSizeY;
        if (indexX<0||indexY<0||indexX>numOfScenesInX-1||indexY>numOfScenesInY-1)
        {
            ++pointView,/* ++pointNormal,*/ ++pointColor, ++pointWeight;
            continue;
        }
        //if (count==5648776)
        //{
        //	std::cout << count << endl;
        //	std::cout << indexX << endl << indexY << endl;
        //}
        if (point->x<boundaryMinXY.x || point->x>boundaryMaxXY.x || point->y<boundaryMinXY.y || point->y>boundaryMaxXY.y)
        {
            ++pointView,/* ++pointNormal,*/ ++pointColor, ++pointWeight;
            continue;
        }
        double remainderX = (point->x - boundaryMinXY.x - sceneSizeX*indexX) / sceneSizeX;
        double remainderY = (point->y - boundaryMinXY.y - sceneSizeY*indexY) / sceneSizeY;

        scenes.at(numOfScenesInX*indexY + indexX).pointcloud.points.push_back(*point);
        scenes.at(numOfScenesInX*indexY + indexX).pointcloud.pointViews.push_back(*pointView);
        scenes.at(numOfScenesInX*indexY + indexX).pointcloud.colors.push_back(*pointColor);
        scenes.at(numOfScenesInX*indexY + indexX).pointcloud.pointWeights.push_back(*pointWeight);
        if (indexX != 0)
        {
            if (remainderX < bufferRange)
            {
                if (remainderY < bufferRange&&indexY != 0)
                {
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX - 1).pointcloud.points.push_back(*point);
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX - 1).pointcloud.pointViews.push_back(*pointView);
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX - 1).pointcloud.colors.push_back(*pointColor);
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX - 1).pointcloud.pointWeights.push_back(*pointWeight);
                }
                else if (remainderY > 1 - bufferRange&&indexY != numOfScenesInY - 1)
                {
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX - 1).pointcloud.points.push_back(*point);
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX - 1).pointcloud.pointViews.push_back(*pointView);
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX - 1).pointcloud.colors.push_back(*pointColor);
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX - 1).pointcloud.pointWeights.push_back(*pointWeight);
                }
                scenes.at(numOfScenesInX*indexY + indexX - 1).pointcloud.points.push_back(*point);
                scenes.at(numOfScenesInX*indexY + indexX - 1).pointcloud.pointViews.push_back(*pointView);
                scenes.at(numOfScenesInX*indexY + indexX - 1).pointcloud.colors.push_back(*pointColor);
                scenes.at(numOfScenesInX*indexY + indexX - 1).pointcloud.pointWeights.push_back(*pointWeight);
            }
        }
        if (indexX != numOfScenesInX - 1)
        {
            if (remainderX > 1 - bufferRange)
            {
                if (remainderY > 1 - bufferRange&&indexY != numOfScenesInY - 1)
                {
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX + 1).pointcloud.points.push_back(*point);
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX + 1).pointcloud.pointViews.push_back(*pointView);
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX + 1).pointcloud.colors.push_back(*pointColor);
                    scenes.at(numOfScenesInX*(indexY + 1) + indexX + 1).pointcloud.pointWeights.push_back(*pointWeight);
                }
                else if (remainderY < bufferRange&&indexY != 0)
                {
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX + 1).pointcloud.points.push_back(*point);
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX + 1).pointcloud.pointViews.push_back(*pointView);
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX + 1).pointcloud.colors.push_back(*pointColor);
                    scenes.at(numOfScenesInX*(indexY - 1) + indexX + 1).pointcloud.pointWeights.push_back(*pointWeight);
                }
                scenes.at(numOfScenesInX*indexY + indexX + 1).pointcloud.points.push_back(*point);
                scenes.at(numOfScenesInX*indexY + indexX + 1).pointcloud.pointViews.push_back(*pointView);
                scenes.at(numOfScenesInX*indexY + indexX + 1).pointcloud.colors.push_back(*pointColor);
                scenes.at(numOfScenesInX*indexY + indexX + 1).pointcloud.pointWeights.push_back(*pointWeight);
            }
        }
        if (indexY != 0)
        {
            if (remainderY < bufferRange)
            {
                scenes.at(numOfScenesInX*(indexY - 1) + indexX).pointcloud.points.push_back(*point);
                scenes.at(numOfScenesInX*(indexY - 1) + indexX).pointcloud.pointViews.push_back(*pointView);
                scenes.at(numOfScenesInX*(indexY - 1) + indexX).pointcloud.colors.push_back(*pointColor);
                scenes.at(numOfScenesInX*(indexY - 1) + indexX).pointcloud.pointWeights.push_back(*pointWeight);
            }
        }
        if (indexY != numOfScenesInY - 1)
        {
            if (remainderY > 1 - bufferRange)
            {
                scenes.at(numOfScenesInX*(indexY + 1) + indexX).pointcloud.points.push_back(*point);
                scenes.at(numOfScenesInX*(indexY + 1) + indexX).pointcloud.pointViews.push_back(*pointView);
                scenes.at(numOfScenesInX*(indexY + 1) + indexX).pointcloud.colors.push_back(*pointColor);
                scenes.at(numOfScenesInX*(indexY + 1) + indexX).pointcloud.pointWeights.push_back(*pointWeight);
            }
        }
        ++pointView,/* ++pointNormal,*/ ++pointColor, ++pointWeight;
    }
    ofstream watcher2("watcher2.txt");
    //for (auto pointView = scenes.at(0).pointcloud.pointViews.begin(); pointView != scenes.at(0).pointcloud.pointViews.end(); pointView++)
    //{
    //	const int viewSize = pointView->size();
    //	for (size_t viewIndex = 0; viewIndex < viewSize; viewIndex++)
    //	{
    //		int index = (*pointView)[viewIndex];
    //		if (index>imageIndexMatcher.at(0).size() - 1)
    //		{
    //			watcher2 << index << " ";
    //		}
    //	}
    //}

    int countView(0);
    for (size_t sceneIndex = 0; sceneIndex < scenes.size(); sceneIndex++)
    {
        for (auto pointView = scenes.at(sceneIndex).pointcloud.pointViews.begin(); pointView != scenes.at(sceneIndex).pointcloud.pointViews.end(); pointView++)
        {
            ++countView;
            const int viewSize = (*pointView).size();
            std::vector<int> viewIndexVec;
            for (size_t viewIndex = 0; viewIndex < viewSize; viewIndex++)
            {
                auto pos = imageIndexMatcher.at(sceneIndex).find(int((*pointView)[viewIndex]));
                if (pos == imageIndexMatcher.at(sceneIndex).end())
                {
                    viewIndexVec.push_back((*pointView)[viewIndex]);
                }
                else
                {
                    //std::cout << imageIndexMatcher.at(sceneIndex).at((*pointView)[viewIndex]) << " ";
                    (*pointView)[viewIndex] = imageIndexMatcher.at(sceneIndex).at((*pointView)[viewIndex]);
                }
            }
            //
            std::sort(viewIndexVec.begin(), viewIndexVec.end());
            for (size_t viewIndex = viewIndexVec.size() - 1; viewIndex > -1; viewIndex--)
            {
                pointView->RemoveAt(viewIndexVec.at(viewIndex));
            }
            for (size_t viewIndex = 0; viewIndex <viewIndexVec.size(); viewIndex++)
            {
                pointView->Remove(viewIndexVec.at(viewIndex));
            }

        }
    }

    ofstream watcher("watcher3.txt");
    countView = 0;
    for (auto pointView = scenes.at(0).pointcloud.pointViews.begin(); pointView != scenes.at(0).pointcloud.pointViews.end(); pointView++)
    {
        const int viewSize = pointView->size();
        for (size_t viewIndex = 0; viewIndex < viewSize; viewIndex++)
        {
            int index = (*pointView)[viewIndex];
            if (index>imageIndexMatcher.at(0).size()-1)
            {
                //std::cout << countView << endl << viewIndex << endl;
                watcher << index << " ";
                //std::cout << index << " ";
                //getchar();
            }
        }
        ++countView;
    }
    std::cout << "passed" << endl;
    //getchar();
    //show image matcher
    //{
    //	for (size_t i = 0; i < scenes.size(); i++)
    //	{
    //		string fileName("matcher_");
    //		fileName += (std::to_string(i) + ".txt");
    //		ofstream writer(fileName);
    //		for (auto index = imageIndexMatcher.at(i).begin(); index != imageIndexMatcher.at(i).end() ; index++)
    //		{
    //			writer << index->first << " " << index->second << endl;
    //		}
    //		writer.close();
    //	}
    //}

    return true;
}


bool SceneDevide::ImageCrop(
        const std::vector<Point2d>& range,
        const std::string & imagePath,
        const double & averageHeight,
        std::map<int, int>& matcher,
        MVS::Scene &scene)
{

    bool writeImageTag(false);
    vector<int> imageIndexToSave;

    const double areaThreshold(2000000);

    vector<Vec4d> groundPointVec(4);
    {
        groundPointVec.at(0)[0] = range.at(0).x; groundPointVec.at(0)[1] = range.at(0).y; groundPointVec.at(0)[2] = averageHeight; groundPointVec.at(0)[3] = 1;
        groundPointVec.at(1)[0] = range.at(1).x; groundPointVec.at(1)[1] = range.at(0).y; groundPointVec.at(1)[2] = averageHeight; groundPointVec.at(1)[3] = 1;
        groundPointVec.at(2)[0] = range.at(1).x; groundPointVec.at(2)[1] = range.at(1).y; groundPointVec.at(2)[2] = averageHeight; groundPointVec.at(2)[3] = 1;
        groundPointVec.at(3)[0] = range.at(0).x; groundPointVec.at(3)[1] = range.at(1).y; groundPointVec.at(3)[2] = averageHeight; groundPointVec.at(3)[3] = 1;
    }

    using boost::geometry::append;
    using boost::geometry::correct;
    using boost::geometry::dsv;

    if (_pScene->images.size() == 0)
    {
        std::cout << "Error: no valid images in scene!" << endl;
        return false;
    }

    Polygon_2d imagePolygon;
    append(imagePolygon, MyPoint{ 0.0, 0.0 });
    append(imagePolygon, MyPoint{ double(imageWidth - 1),0.0 });
    append(imagePolygon, MyPoint{ double(imageWidth - 1),double(imageHeight - 1) });
    append(imagePolygon, MyPoint{ 0.0,double(imageHeight - 1) });
    correct(imagePolygon);

    for (size_t imageIndex = 0; imageIndex < _pScene->images.size(); imageIndex++)
    {
        vector<Vec3d> imagePointHVec(4);
        vector<MyPoint> imagePointVec;
        Image imageIndexed = _pScene->images[imageIndex]; //FIXME: this copy opeartion seems a big cost

        //update the camera and compose the project matrix

        imageIndexed.width = imageWidth;
        imageIndexed.height = imageHeight;
        //std::cout << imageIndexed.scale << endl;
        //std::cout << imageIndexed.width << endl << endl;
        //getchar();
        imageIndexed.UpdateCamera(_pScene->platforms);

        for (size_t i = 0; i < groundPointVec.size(); i++)
        {
            imagePointHVec.at(i) = imageIndexed.camera.P*groundPointVec.at(i);
            imagePointVec.push_back(MyPoint(imagePointHVec.at(i)[0] / imagePointHVec.at(i)[2], imagePointHVec.at(i)[1] / imagePointHVec.at(i)[2]));
        }

        Polygon_2d polygon;
        append(polygon, imagePointVec.at(0));
        append(polygon, imagePointVec.at(1));
        append(polygon, imagePointVec.at(2));
        append(polygon, imagePointVec.at(3));
        correct(polygon);

        std::vector<Polygon_2d> polys;
        if (boost::geometry::intersection(imagePolygon, polygon, polys))
        {
            if (polys.size() == 0)
            {
                continue;
            }

            Polygon_2d overlap = polys.at(0);
            if (boost::geometry::area(overlap) < areaThreshold)
            {
                continue;
            }

            vector<double> xVec, yVec;
            for (size_t i = 0; i < overlap.outer().size(); i++)
            {
                xVec.push_back(overlap.outer().at(i).x);
                yVec.push_back(overlap.outer().at(i).y);
            }
            int minX = *min_element(xVec.begin(), xVec.end());
            int minY = *min_element(yVec.begin(), yVec.end());
            int maxX = *max_element(xVec.begin(), xVec.end());
            int maxY = *max_element(yVec.begin(), yVec.end());

            //crop the image and update the images in the scene
            double xo = imageIndexed.camera.K(0, 2);
            double yo = imageIndexed.camera.K(1, 2);
            if ((maxX - minX) < 1000 || (maxY - minY) < 1000)
            {
                continue;
            }

            string imageName = string("/home/jxq/CODE/image") + imageIndexed.name; //FIXME: image path specific
            string imageOutputName = imagePath + "/" + imageName.substr(imageName.find_last_of('/'), imageName.length() - imageName.find_last_of('/'));
            imageIndexToSave.push_back(imageIndex);
            matcher.insert(pair<int, int>(imageIndex, imageIndexToSave.size() - 1));
            bool folder_exists(const std::string& folder);
//            if (writeImageTag)
//            {
//                //std::cout << imageName << endl;
//                //getchar();
//                cv::Mat image = cv::imread(imageName);
//                //
//                //std::cout << minX << endl << minY << endl << maxX << endl << maxY << endl;
//                //
//                cv::Mat subImage = image(cv::Rect(minX, minY, maxX - minX, maxY - minY));
//                //creat output path
//                if (!stlplus::folder_exists(imagePath + "/"))
//                {
//                    if (!stlplus::folder_create(imagePath + "/"))
//                    {
//                        std::cerr << "\nCannot create output directory " << imagePath + "/" << std::endl;
//                        return false;
//                    }
//                }
//                cv::imwrite(imageOutputName, subImage);
//            }

            //std::cout << imageIndexed.name << " " << minX << " " << minY << endl;
            //std::cout << _pScene->platforms[imageIndexed.platformID].cameras[imageIndexed.cameraID].K(0, 2);

            double valueMax = maxX - minX > maxY - minY ? maxX - minX : maxY - minY;
            double deltX = (xo - minX) / valueMax - _pScene->platforms[imageIndexed.platformID].cameras[imageIndexed.cameraID].K(0, 2);
            double deltY = (yo - minY) / valueMax - _pScene->platforms[imageIndexed.platformID].cameras[imageIndexed.cameraID].K(1, 2);
            //double deltX = - minX / imageWidth ;
            //double deltY = - minY / imageWidth;


            //std::cout << imageIndexed.name << endl << imageIndexed.platformID << endl << imageIndexed.cameraID << endl; getchar();
            scene.platforms[imageIndexed.platformID].cameras[imageIndexed.cameraID].UpdatePrincipalPoint(Point2(deltX, deltY));

            //cout << scene.platforms[imageIndexed.platformID].cameras[imageIndexed.cameraID].K << endl; getchar();
            double focalOld = _pScene->platforms[imageIndexed.platformID].cameras[imageIndexed.cameraID].K(0, 0);
            double focalNew = focalOld*imageWidth / valueMax;	// FIXME: wether this expression hold for image which height is bigger than width
            scene.platforms[imageIndexed.platformID].cameras[imageIndexed.cameraID].UpdateFocalLengthAbs(focalNew);

            imageIndexed.width = maxX - minX;
            imageIndexed.height = maxY - minY;
            imageIndexed.UpdateCamera(scene.platforms);
            imageIndexed.name = imageOutputName;
            //imageIndexed.camera.ComposeP();

            //add image to the scene
            scene.images.push_back(imageIndexed);
        }
        else
        {
            continue;
        }
    }

    //update image's neighbor image
    int imageIndex(0);
    for (auto imageIndexed = scene.images.begin(); imageIndexed != scene.images.end(); imageIndexed++, imageIndex++)
    {
        const int neighborSize = imageIndexed->neighbors.size();
        vector<MVS::ViewScore> imageIndexToRemove;
        for (size_t neighborIndex = 0; neighborIndex < neighborSize; neighborIndex++)
        {
            auto &neighbor = imageIndexed->neighbors[neighborIndex];
            auto pos = matcher.find(neighbor.idx.ID);
            if (pos == matcher.end())
            {
                imageIndexToRemove.push_back(neighbor);
                //imageIndexed->neighbors.RemoveAt(neighborIndex);
            }
            else
            {
                //std::cout << _pScene->images[imageIndexed->neighbors[neighborIndex].idx.ID].name << endl
                //	<< scene.images[matcher.at(imageIndexed->neighbors[neighborIndex].idx.ID)].name << endl;
                neighbor.idx.ID = matcher.at(neighbor.idx.ID);
                //getchar();
            }
        }
        //std::sort(imageIndexToRemove.begin(), imageIndexToRemove.end());
        //for (size_t i = imageIndexToRemove.size()-1; i >-1; i--)
        //{
        //	imageIndexed->neighbors.RemoveAt(imageIndexToRemove.at(i));
        //}
        for (size_t i = 0; i < imageIndexToRemove.size(); i++)
        {
            imageIndexed->neighbors.Remove(imageIndexToRemove.at(i));
        }

    }
    return true;
}

bool SceneDevide::PointCloudCrop(const std::vector<Point2d>& range, std::map<int, int>& matcher, MVS::Scene & scene)
{
    if (_pScene->pointcloud.GetSize() == 0)
    {
        std::cout << "Error: Enmpty point cloud in scene" << endl;
        return false;
    }

    std::cout << endl << _pScene->pointcloud.points.size() << endl
              << _pScene->pointcloud.pointViews.size() << endl
              << _pScene->pointcloud.normals.size() << endl
              << _pScene->pointcloud.colors.size() << endl
              << _pScene->pointcloud.pointWeights.size() << endl << endl;

    if (_pScene->pointcloud.points.size() != _pScene->pointcloud.pointViews.size() ||
        //_pScene->pointcloud.points.size() != _pScene->pointcloud.normals.size() ||
        _pScene->pointcloud.points.size() != _pScene->pointcloud.colors.size() ||
        _pScene->pointcloud.points.size() != _pScene->pointcloud.pointWeights.size())
    {
        std::cout << "Error: Invalid point cloud in scene" << endl;
        return false;
    }

    //FIXME: compare which is faster iterater or [] operation
    auto pointView = _pScene->pointcloud.pointViews.begin();
    //auto pointNormal = _pScene->pointcloud.normals.begin();
    auto pointColor = _pScene->pointcloud.colors.begin();
    auto pointWeight = _pScene->pointcloud.pointWeights.begin();
    for (auto point = _pScene->pointcloud.points.begin(); point != _pScene->pointcloud.points.end(); ++point)
    {
        if (point->x > range.at(0).x&&
            point->x < range.at(1).x&&
            point->y > range.at(0).y&&
            point->y < range.at(1).y)
        {
            scene.pointcloud.points.push_back(*point);
            scene.pointcloud.pointViews.push_back(*pointView);
            //scene.pointcloud.normals.push_back(*pointNormal);
            scene.pointcloud.colors.push_back(*pointColor);
            scene.pointcloud.pointWeights.push_back(*pointWeight);
        }
        ++pointView,/* ++pointNormal,*/ ++pointColor, ++pointWeight;

    }
    //for (auto pointView = scene.pointcloud.pointViews.begin(); pointView != scene.pointcloud.pointViews.end(); pointView++)
    //{
    //	const int viewSize = pointView->size();
    //	std::vector<int> viewIndexVec;
    //	for (size_t viewIndex = 0; viewIndex < viewSize; viewIndex++)
    //	{
    //		auto pos = matcher.find(viewIndex);
    //		if (pos == matcher.end())
    //		{
    //			viewIndexVec.push_back(viewIndex);
    //		}
    //		else
    //		{
    //			//std::cout << (*pointView)[viewIndex]; getchar();
    //			(*pointView)[viewIndex] = matcher.at((*pointView)[viewIndex]);
    //		}
    //	}
    //	//
    //	for (size_t viewIndex = viewIndexVec.size() - 1; viewIndex > -1; viewIndex--)
    //	{
    //		pointView->RemoveAt(viewIndexVec.at(viewIndex));
    //	}
    //}
    return true;
}