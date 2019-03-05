/*
* Common.h
*/

#ifndef _MVS_COMMON_H_
#define _MVS_COMMON_H_


// I N C L U D E S /////////////////////////////////////////////////

#if defined(MVS_EXPORTS) && !defined(Common_EXPORTS)
#define Common_EXPORTS
#endif

#include "../Common/Common.h"
#include "../IO/Common.h"
#include "../Math/Common.h"

#ifndef MVS_API
#define MVS_API GENERAL_API
#endif
#ifndef MVS_TPL
#define MVS_TPL GENERAL_TPL
#endif


// D E F I N E S ///////////////////////////////////////////////////


// P R O T O T Y P E S /////////////////////////////////////////////

using namespace SEACAVE;

namespace MVS {

/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_COMMON_H_
