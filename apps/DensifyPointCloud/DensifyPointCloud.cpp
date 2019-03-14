/*
 * DensifyPointCloud.cpp
 */

#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#include <boost/program_options.hpp>
#include <iostream>

using namespace std;
using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

#define APPNAME _T("DensifyPointCloud")


// S T R U C T S ///////////////////////////////////////////////////

namespace OPT {
String strInputFileName;
String strOutputFileName;
String strMeshFileName;
String strDenseConfigFileName;
float fSampleMesh;
int nArchiveType;
int nProcessPriority;
unsigned nMaxThreads;
String strConfigFileName;
boost::program_options::variables_map vm;
} // namespace OPT

// initialize and parse the command line parameters
bool Initialize(size_t argc, LPCTSTR* argv)
{
	// initialize log and console
	OPEN_LOG();
	OPEN_LOGCONSOLE();

	// group of options allowed only on command line
	boost::program_options::options_description generic("Generic options");
	generic.add_options()
		("help,h", "produce this help message")
		("working-folder,w", boost::program_options::value<std::string>(&WORKING_FOLDER), "working directory (default current directory)")
		("config-file,c", boost::program_options::value<std::string>(&OPT::strConfigFileName)->default_value(APPNAME _T(".cfg")), "file name containing program options")
		("archive-type", boost::program_options::value(&OPT::nArchiveType)->default_value(2), "project archive type: 0-text, 1-binary, 2-compressed binary")
		("process-priority", boost::program_options::value(&OPT::nProcessPriority)->default_value(-1), "process priority (below normal by default)")
		("max-threads", boost::program_options::value(&OPT::nMaxThreads)->default_value(0), "maximum number of threads (0 for using all available cores)")
		#if TD_VERBOSE != TD_VERBOSE_OFF
		("verbosity,v", boost::program_options::value(&g_nVerbosityLevel)->default_value(
			#if TD_VERBOSE == TD_VERBOSE_DEBUG
			3
			#else
			2
			#endif
			), "verbosity level")
		#endif
		;

	// group of options allowed both on command line and in config file
	// 命令行和配置文件中都允许的一组选项
	unsigned nResolutionLevel;
	unsigned nMinResolution;
	unsigned nNumViews;
	unsigned nMinViewsFuse;
	unsigned nEstimateColors;
	unsigned nEstimateNormals;
	boost::program_options::options_description config("Densify options");
	config.add_options()
		("input-file,i", boost::program_options::value<std::string>(&OPT::strInputFileName), "input filename containing camera poses and image list")
		("output-file,o", boost::program_options::value<std::string>(&OPT::strOutputFileName), "output filename for storing the dense point-cloud")
		("resolution-level", boost::program_options::value<unsigned>(&nResolutionLevel)->default_value(1), "how many times to scale down the images before point cloud computation")
		("min-resolution", boost::program_options::value<unsigned>(&nMinResolution)->default_value(640), "do not scale images lower than this resolution")
		("number-views", boost::program_options::value<unsigned>(&nNumViews)->default_value(4), "number of views used for depth-map estimation (0 - all neighbor views available)")
		("number-views-fuse", boost::program_options::value<unsigned>(&nMinViewsFuse)->default_value(3), "minimum number of images that agrees with an estimate during fusion in order to consider it inlier")
		("estimate-colors", boost::program_options::value<unsigned>(&nEstimateColors)->default_value(1), "estimate the colors for the dense point-cloud")
		("estimate-normals", boost::program_options::value<unsigned>(&nEstimateNormals)->default_value(0), "estimate the normals for the dense point-cloud")
		("sample-mesh", boost::program_options::value<float>(&OPT::fSampleMesh)->default_value(0.f), "uniformly samples points on a mesh (0 - disabled, <0 - number of points, >0 - sample density per square unit)")
		;

	// hidden options, allowed both on command line and
	// in config file, but will not be shown to the user
	boost::program_options::options_description hidden("Hidden options");
	hidden.add_options()
		("dense-config-file", boost::program_options::value<std::string>(&OPT::strDenseConfigFileName), "optional configuration file for the densifier (overwritten by the command line options)")
		;

	boost::program_options::options_description cmdline_options;
	cmdline_options.add(generic).add(config).add(hidden);

	boost::program_options::options_description config_file_options;
	config_file_options.add(config).add(hidden);

	boost::program_options::positional_options_description p;
	p.add("input-file", -1);

	try {
		// parse command line options
		boost::program_options::store(boost::program_options::command_line_parser((int)argc, argv).options(cmdline_options).positional(p).run(), OPT::vm);
		boost::program_options::notify(OPT::vm);
		INIT_WORKING_FOLDER;
		// parse configuration file
		std::ifstream ifs(MAKE_PATH_SAFE(OPT::strConfigFileName));
		if (ifs) {
			boost::program_options::store(parse_config_file(ifs, config_file_options), OPT::vm);
			boost::program_options::notify(OPT::vm);
		}
	}
	catch (const std::exception& e) {
		LOG(e.what());
		return false;
	}

	// initialize the log file
	OPEN_LOGFILE(MAKE_PATH(APPNAME _T("-")+Util::getUniqueName(0)+_T(".log")).c_str());

	// print application details: version and command line
	Util::LogBuild();
	LOG(_T("Command line:%s"), Util::CommandLineToString(argc, argv).c_str());

	// validate input
	Util::ensureValidPath(OPT::strInputFileName);
	Util::ensureUnifySlash(OPT::strInputFileName);
	if (OPT::vm.count("help") || OPT::strInputFileName.IsEmpty()) {
		boost::program_options::options_description visible("Available options");
		visible.add(generic).add(config);
		GET_LOG() << visible;
	}
	if (OPT::strInputFileName.IsEmpty())
		return false;

	// initialize optional options
	Util::ensureValidPath(OPT::strOutputFileName);
	Util::ensureUnifySlash(OPT::strOutputFileName);
	if (OPT::strOutputFileName.IsEmpty())
		OPT::strOutputFileName = Util::getFileFullName(OPT::strInputFileName) + _T("_dense.mvs");

	// init dense options
	OPTDENSE::init();
	const bool bValidConfig(OPTDENSE::oConfig.Load(OPT::strDenseConfigFileName));
	OPTDENSE::update();
	OPTDENSE::nResolutionLevel = nResolutionLevel;
	OPTDENSE::nMinResolution = nMinResolution;
	OPTDENSE::nNumViews = nNumViews;
	OPTDENSE::nMinViewsFuse = nMinViewsFuse;
	OPTDENSE::nEstimateColors = nEstimateColors;
	OPTDENSE::nEstimateNormals = nEstimateNormals;
	if (!bValidConfig)
		OPTDENSE::oConfig.Save(OPT::strDenseConfigFileName);

	// initialize global options
	Process::setCurrentProcessPriority((Process::Priority)OPT::nProcessPriority);
	#ifdef _USE_OPENMP
	if (OPT::nMaxThreads != 0)
		omp_set_num_threads(OPT::nMaxThreads);
	#endif

	#ifdef _USE_BREAKPAD
	// start memory dumper
	MiniDumper::Create(APPNAME, WORKING_FOLDER);
	#endif
	return true;
}

// finalize application instance
void Finalize()
{
	#if TD_VERBOSE != TD_VERBOSE_OFF
	// print memory statistics
	Util::LogMemoryInfo();
	#endif

	CLOSE_LOGFILE();
	CLOSE_LOGCONSOLE();
	CLOSE_LOG();
}

int main(int argc, LPCTSTR* argv)
{
	#ifdef _DEBUGINFO
	// set _crtBreakAlloc index to stop in <dbgheap.c> at allocation
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);// | _CRTDBG_CHECK_ALWAYS_DF);
	#endif

	// 初始化
	if (!Initialize(argc, argv))
		return EXIT_FAILURE;

	Scene scene(OPT::nMaxThreads);

	/*// (0 - 禁用， <0 - 点数， >0 - 每平方单位的样本密度） TODO
	if (OPT::fSampleMesh != 0) {	// 均匀采样网格上的点
	    cout << 1 << endl;
		// sample input mesh and export the obtained point-cloud	采样输入网格并导出获得的点云
		if (!scene.mesh.Load(MAKE_PATH_SAFE(OPT::strInputFileName)))    // 从obj或者ply文件中导入网格
			return EXIT_FAILURE;
		TD_TIMER_START();
		PointCloud pointcloud;
		if (OPT::fSampleMesh > 0) {
            cout << 2 << endl;
            scene.mesh.SamplePoints(OPT::fSampleMesh, 0, pointcloud);
        }
		else {
            cout << 3 << endl;
            scene.mesh.SamplePoints((unsigned) ROUND2INT(-OPT::fSampleMesh), pointcloud);
        }
		VERBOSE("(apps/DensifyPointCloud/DensifyPointCloud.cpp)Sample mesh completed: %u points (%s)", pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
		pointcloud.Save(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName))+_T(".ply"));
		Finalize();
		return EXIT_SUCCESS;
	}*/

	// 加载并估计稠密点云
	if (!scene.Load(MAKE_PATH_SAFE(OPT::strInputFileName)))	// 初始化scene类
		return EXIT_FAILURE;
	if (scene.pointcloud.IsEmpty()) {	// 点云不能为空，确保存在稀疏点云
		VERBOSE("(apps/DensifyPointCloud/DensifyPointCloud.cpp)error: empty initial point-cloud");
		return EXIT_FAILURE;
	}

	// 正文 算法
	// nArchiveType project archive type: 0-text, 1-binary, 2-compressed binary  项目存档类型
	// ARCHIVE_MVS -1
	// cout << (ARCHIVE_TYPE)OPT::nArchiveType << endl;  // 2
	if ((ARCHIVE_TYPE)OPT::nArchiveType != ARCHIVE_MVS) {
		TD_TIMER_START();
		if (!scene.DenseReconstruction())
			return EXIT_FAILURE;
		VERBOSE("(apps/DensifyPointCloud/DensifyPointCloud.cpp)Densifying point-cloud completed: %u points (%s)", scene.pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// 保存最终网格
	const String baseFileName(MAKE_PATH_SAFE(Util::getFileFullName(OPT::strOutputFileName)));
	scene.Save(baseFileName+_T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
	scene.pointcloud.Save(baseFileName+_T(".ply"));
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2)
		scene.ExportCamerasMLP(baseFileName+_T(".mlp"), baseFileName+_T(".ply"));
	#endif

	Finalize();
	return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/
