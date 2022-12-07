/************************************************************************/
/* Lottery Vission Tool                                                 */
/*                                                                      */
/* Config.h                                                             */
/*                                                                      */
/* Nondas Masalis <bestterminalsgr@outlook.com>                         */
/*                                                                      */
/* Config.h                                                             */
/*                                                                      */
/************************************************************************/

#ifndef CONFIG_H
#define CONFIG_H

// Qt
#include <QThread>

#undef USE_OPENCV
#undef USE_ESCAPI
#define USE_FFMPEG_LIB

#define TABLET_RELEASE
#undef HP_LAPTOP
#undef DESKTOP_WINDOWS
#undef DESKTOP_LINUX
#undef DESKTOP_MAC


const int POWERBALL_PLAYSLIP_ID = 5145;
const int KENO_PLAYSLIP_ID      = 1105;


#if defined TABLET_RELEASE
static const QString CAMERA_NAME = QStringLiteral("ManyCam Virtual Webcam");//QStringLiteral("Microsoft LifeCam HD-3000");//QStringLiteral("ManyCam Virtual Webcam");//QStringLiteral("AR0543");//QStringLiteral("OV5648");
const double FPS = 15.;
#elif defined HP_LAPTOP
static const QString CAMERA_NAME = QStringLiteral("HP HD Webcam");
const double FPS = 15.;
#elif defined DESKTOP_WINDOWS || defined DESKTOP_LINUX
static const QString CAMERA_NAME = QStringLiteral("Microsoft LifeCam HD-3000");
const double FPS = 15.;
#endif

// FPS statistics queue lengths
#define CAPTURE_FPS_STAT_QUEUE_LENGTH       32

const quint16 RESOLUTION_WIDTH = 1280;//1920;//1280;
#if defined USE_ESCAPI
const quint16 RESOLUTION_HEIGHT = 960;
#else
const quint16 RESOLUTION_HEIGHT = 720;//1080;//720;
#endif


/*
QThread::IdlePriority	        0	scheduled only when no other threads are running.
QThread::LowestPriority	        1	scheduled less often than LowPriority.
QThread::LowPriority        	2	scheduled less often than NormalPriority.
QThread::NormalPriority	        3	the default priority of the operating system.
QThread::HighPriority	        4	scheduled more often than NormalPriority.
QThread::HighestPriority	    5	scheduled more often than HighPriority.
QThread::TimeCriticalPriority	6	scheduled as often as possible.
QThread::InheritPriority	    7	use the same priority as the creating thread. This is the default.
*/
const QThread::Priority DEFAULT_CAP_THREAD_PRIO    = QThread::HighPriority;
const QThread::Priority DEFAULT_MOTION_THREAD_PRIO = QThread::HighPriority;
const QThread::Priority DEFAULT_PROC_THREAD_PRIO   = QThread::NormalPriority;

// Image buffer size
const qint32 DEFAULT_IMAGE_BUFFER_SIZE = 64; // size does not really matter

#undef TEST_IDEAL_IMAGE
#undef PREVIEW_MARKS

#endif // CONFIG_H
