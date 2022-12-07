#ifndef PATTERNDETECTION_H
#define PATTERNDETECTION_H

#include <QString>
#include <QObject>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include "definitions.h"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"

using namespace std;
using namespace cv;

class QSoundEffect;

//Base class
class patternDetection : public QObject
{
	Q_OBJECT

public:

	patternDetection ( QObject* parent = 0 );
	~patternDetection();

	QList<int> processCoupon ( Mat mImage, quint32 iPlaySlipId );

private:
	int errorCode=0;

	bool m_bSpeedUpDetection;
	bool m_bPreviewProcessed;
	int m_iEpsilon;
	bool m_bSaveImage;
	QMutex m_Lock;
	float m_xDistance = 0.0;
	const Point m_Center {0,0};


	Mat perspectTransform ( Mat image );
	vector<Point> detectBoxes ( Mat image , quint32 iNumberOfDefinedCells );
	QList<int> ROIaveragePixelValue ( Mat image, vector<Point> boxCenters );

	void extentLargestSquare ( vector <Point>& largestSquare );
	double calculateAngle ( Point pt1, Point pt2, Point pt0 );
	vector<Point> interpolateBoxes ( vector<vector<Point>> foundBoxes );
	void sortCorners ( vector<Point>& corners, Point center );
	void sortBoxes ( vector<Point>& corners );
	Point detectCenters ( vector<Point>& corners );
	QSoundEffect* m_pcSound;
	QSoundEffect* m_pcSoundError;
};

#endif // PATTERNDETECTION_H

