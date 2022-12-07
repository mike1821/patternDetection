#include "patternDetection.h"
#include "definitions.h"
#include "MatToQImage.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <QDir>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/timeb.h>
#include <time.h>
#include <QTime>
#include <QtDebug>
//#include "ResourceManager.h"
#include <QSoundEffect>

using namespace cv;
using namespace std;

int counter=0;
void Log(std::string message);

///////////////////////////////////////////////////////////////////////////////////

patternDetection::patternDetection (  QObject* parent )
	: QObject ( parent )
{

	m_pcSound = new QSoundEffect (this);
	m_pcSound->setSource(QUrl::fromLocalFile(QString("CameraFocusBeep.wav")));

	m_bSpeedUpDetection = false;
	m_bPreviewProcessed = false;
	m_iEpsilon = 8;

    /*QObject::connect( ResourceManager::getResourceManagerInstance()->lotteryVisionConfig(),
					  &LotteryVisionConfig::speedUpDetectionChanged,
					  this, [this](bool bNewVal)
	{
		QMutexLocker locker(&m_Lock);
		m_bSpeedUpDetection = bNewVal;
		qDebug () << "\n\nOptimization is enabled: " << m_bSpeedUpDetection << "\n\n";

	});

	QObject::connect( ResourceManager::getResourceManagerInstance()->lotteryVisionConfig(),
					  &LotteryVisionConfig::previewProcessedChanged,
					  this, [this](bool bNewVal)
	{
		QMutexLocker locker(&m_Lock);
		m_bPreviewProcessed = bNewVal;
		qDebug () << "\n\nPreview in openCV is enabled: " << m_bPreviewProcessed << "\n\n";
	});

	QObject::connect( ResourceManager::getResourceManagerInstance()->lotteryVisionConfig(),
					  &LotteryVisionConfig::epsilonChanged,
					  this, [this](int iNewVal)
	{
		QMutexLocker locker(&m_Lock);
		m_iEpsilon = iNewVal;
		qDebug () << "\n\nEpsilon set to: " << m_iEpsilon << "\n\n";
	});

	QObject::connect( ResourceManager::getResourceManagerInstance()->lotteryVisionConfig(),
					  &LotteryVisionConfig::saveImageChanged,
					  this, [this](bool bNewVal)
	{
		QMutexLocker locker(&m_Lock);
		m_bSaveImage = bNewVal;
		qDebug () << "\n\nSave Image flag in (PR) is enabled: " << m_bSaveImage << "\n\n";
    });*/


}

patternDetection::~patternDetection(){

}

double patternDetection::calculateAngle( Point pt1, Point pt2, Point pt0 )
{
	// Finds a cosine of angle between vectors
	// from pt0->pt1 and from pt0->pt2
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


/**************************************************************
* Input: Mat qsImage
* Output: QString marks
*
* Description: Process input color image and return the detected
* bet marks of the specified coupon
* ************************************************************/
QList<int> patternDetection::processCoupon( Mat mImage )
{
	vector<Point> boxCenters;
	QList<int> detectedMarks = QList<int>();

	int iTotalTime = 0;
	int iMidTime   = 0;

	Mat croppedImage;

	qDebug () << "----------------> Processing image";

	QTime t;
	iTotalTime += t.restart();

	croppedImage = perspectTransform ( mImage );

	iMidTime = t.restart();
	iTotalTime += iMidTime;
	qDebug () << "PerspectiveTransform finished in: " << iMidTime;


	// Check whether the image was cropped and if not beep twice and return.
	if ( !croppedImage.data ) {

		qDebug () << "Perspective transformation failed";

		if ( m_bSaveImage ) {
			QString qsFileName = QString("C:\\temp\\PRimage-%1.jpg").arg(QDateTime::currentDateTime().toTime_t());
			imwrite(qPrintable(qsFileName),mImage);
		}

		// m_pcSound->setLoopCount(3);
		// m_pcSound->play();

		return detectedMarks;
	}


	boxCenters = detectBoxes ( croppedImage ); //Detect boxes on the cropped image

    FILE *fp;
    fp=fopen("C:\\temp\\result.txt","w+");
    if(fp)
    {
        for(size_t i=0; i<boxCenters.size();i++)
            fprintf(fp,"%d - %d\n", boxCenters[i].x, boxCenters[i].y);
        fclose(fp);
    }

	iMidTime = t.restart();
	iTotalTime += iMidTime;
	qDebug () << "detectBoxes finished in: " << iMidTime;


	// Check whether all boxes for powerball playslip were found. If not beep twice.
	if ( boxCenters.size() == POWERBALL ) {

		detectedMarks = ROIaveragePixelValue ( croppedImage, boxCenters ); //Detect avgPixel value on each box
		iMidTime = t.restart();
		iTotalTime += iMidTime;
		qDebug () << "ROIaveragePixelValue finished in: " << iMidTime;

		qDebug () << "Number of marks found: " << detectedMarks.size () << " Detected marks: " << detectedMarks;

	} else {

		// m_pcSound->setLoopCount(2);
		// m_pcSound->play();
		qDebug () << "\n\nFound: " << boxCenters.size() << " out of: " << POWERBALL << " Marks detection was aborted!\n\n";

	}

	qDebug () << "----------------> Processing time: " << iTotalTime << " ms\n\n";

	return detectedMarks;
}

/**
 * @sa extentLargestSquare
 * @param largestSquare
 * @return largestSquare
 * @brief extends the cropped rectangle by some extend to make sure boxes are always in picture
 */
void patternDetection::extentLargestSquare ( vector <Point>& largestSquare )
{
	/* Sorted Corners

	  [0] ...... [1]
	   .          .
	   .          .
	  [3] ...... [2]

	*/

	// Top Right X
	int iMargin = FRAME_WIDTH - largestSquare[TOP_RIGHT].x;

	if ( CROP_EXTEND_X >= iMargin ) {

		largestSquare[TOP_RIGHT].x += iMargin;

	} else {

		largestSquare[TOP_RIGHT].x += CROP_EXTEND_X;
	}

	// Bottom Right X
	iMargin = FRAME_WIDTH - largestSquare[BOTTOM_RIGHT].x;

	if ( CROP_EXTEND_X >= iMargin ) {

		 largestSquare[BOTTOM_RIGHT].x += iMargin;

	} else {

		largestSquare[BOTTOM_RIGHT].x += CROP_EXTEND_X;
	}

	// Top Left Y
	if ( largestSquare[TOP_LEFT].y >= CROP_EXTEND_Y ) {

		 largestSquare[TOP_LEFT].y -= CROP_EXTEND_Y;

	} else {

		largestSquare[TOP_LEFT].y -= largestSquare[TOP_LEFT].y;
	}

	// Top Right Y
	if ( largestSquare[TOP_RIGHT].y >= CROP_EXTEND_Y ) {

		 largestSquare[TOP_RIGHT].y -= CROP_EXTEND_Y;

	} else {

		largestSquare[TOP_RIGHT].y -= largestSquare[TOP_RIGHT].y;
	}

	// Bottom Left Y
	iMargin = FRAME_HEIGHT - largestSquare[BOTTOM_LEFT].y;

	if ( CROP_EXTEND_Y >= iMargin ) {

		 largestSquare[BOTTOM_LEFT].y += iMargin;

	} else {

		largestSquare[BOTTOM_LEFT].y += CROP_EXTEND_Y;
	}

	// Bottom Right Y
	iMargin = FRAME_HEIGHT - largestSquare[BOTTOM_RIGHT].y;

	if ( CROP_EXTEND_Y >= iMargin ) {

		 largestSquare[BOTTOM_RIGHT].y += iMargin;

	} else {

		largestSquare[BOTTOM_RIGHT].y += CROP_EXTEND_Y;
	}
}

/**
 * @sa perspectTransform
 * @param image
 * @param direction
 * @return cropped image transormed arround the outer box
 */
cv::Mat patternDetection::perspectTransform ( Mat image )
{
	vector<vector <Point>> contours;
	vector<vector <Point>> squares;
	vector<Point> approx;

	quint32 iLargestContourIndex = 0;
	double dLargestArea          = 0.;
	bool bManualAdustment        = false;

	Mat gray, pyr, timg, croppedImage;


	// Debug you own pictures.
	//	image = imread("C://temp//image-0006.jpg");
	//	image = imread("C://temp//image-0020.jpg");

	if ( image.empty() ) {
		qDebug () << "Provided image is empty";
		return croppedImage;
	}


	// down-scale and upscale the image to filter out the noise. When we do so the number
	// of contours reduces substantially resulting in faster code because we got a smaller
	// contours vector to deal with.
	if ( m_bSpeedUpDetection ) {

		pyrDown ( Mat(image), pyr);
		pyrUp   ( pyr, timg );

	} else {

		timg = image.clone ();
	}

	// There are a lot of conditions where just a pair of values for blockSize and C in adaptiveThreshold will not
	// always produce an outer rectangle so we can crop the image. So we'll try several values.
	QList<QPair<int,int>> lThresholdValues = QList<QPair<int,int>>() << qMakePair(11,2) << qMakePair(9,5) << qMakePair(75,10) << qMakePair(91,7) << qMakePair(161,3) << qMakePair(181,15) << qMakePair(101,-4);

	for ( int iRetry = 0; iRetry < lThresholdValues.size(); ++iRetry ) {

		cvtColor ( timg, gray, CV_BGR2GRAY);
		adaptiveThreshold ( gray, gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, lThresholdValues.at(iRetry).first, lThresholdValues.at(iRetry).second );

		// Find the contours in the image
		findContours( gray.clone(), contours, RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS );

		qDebug () << "contours size: " << contours.size();

		// test each contour
		for ( size_t i = 0; i < contours.size(); i++ ) {

			approx.clear();

			// approximate contour with accuracy proportional to the contour perimeter
			// 32 is a magic number (epsilon) which I really didn't look it in depth to see how it works in approxPolyDP method
			// A vaule of 13 is also good as well as a value of: arcLength( Mat(contours[i]), true)*0.02
			approxPolyDP( Mat(contours[i]), approx, 32, true);

			// square contours should have 4 vertices after approximation
			// relatively large area (to filter out noisy contours)
			// and be convex.
			// Note: absolute value of an area is used because
			// area may be positive or negative - in accordance with the
			// contour orientation

			if ( approx.size() == 4
				 && fabs(contourArea(Mat(approx))) > AREA_MIN_SIZE
				 && fabs(contourArea(Mat(approx))) < AREA_MAX_SIZE
				 && isContourConvex(Mat(approx)) )
			{
				double maxCosine = 0;

				for( int j = 2; j < 5; j++ )
				{
					// find the maximum cosine of the angle between joint edges
					double cosine = fabs(calculateAngle ( approx[j%4], approx[j-2], approx[j-1] ) );
					maxCosine = MAX(maxCosine, cosine);
				}

				// if cosines of all angles are small (all angles are ~90 degree) then write quandrange
				// vertices to resultant sequence

				if( maxCosine < 0.3 ) {

					squares.push_back(approx);
				}
			}
		}

		// If we got to retry values beyond 75,10 then it means the photo is contaminated with a lot of noise
		// and there is the possibility to crop valuable space from boxes areas. So, we'll manual allow some
		// margin when do the crop later on.
		if ( iRetry > 0 ) {

			bManualAdustment = true;
		}

		if ( squares.size() != 0 ) {
			// We've found what we wanted, drop out
			break;
		}
	}


	// Here we couldn't find any squares from contours based on AREA_MIN_SIZE and AREA_MAX_SIZE
	// requirements. So....we'll see what we'll do.
	if ( squares.size() == 0 ) {
		qDebug () << "Could not big square.";
		return croppedImage;
	}

	// Sort detected boxes
	for (auto &vp: squares) {
		blowMe (vp);
	}

	double dArea = 0.;
	for ( size_t iSquaresIdx = 0; iSquaresIdx < squares.size(); ++iSquaresIdx ) {

		dArea = contourArea( squares[iSquaresIdx] );  //  Find the area of contour

		if ( dArea > dLargestArea && dArea < AREA_MAX_SIZE ) {

			// There are might cases where we got very similar areas of two outer boxes
			// as a result of stacking playslips. In that case, we consider the smaller
			// area withing the threshold as a valid one. e.g, area1 = 400000, and area2 = 430000
			// we will be interested in area1. This is what our image will be cropped to.
			if ( dArea - dLargestArea > AREA_THRESHOLD ) {

				dLargestArea = dArea;
				iLargestContourIndex = iSquaresIdx;
			}
			qDebug () << "dLargestArea: " << dLargestArea << " index: " << iSquaresIdx;
		}
	}

	extentLargestSquare ( squares[iLargestContourIndex] );


	// OK we got the area we are interested in, approximate the square now.
	vector<vector<Point> > contours_poly(1);
	approxPolyDP ( Mat( squares [iLargestContourIndex] ), contours_poly[0], 8, true );
	Rect boundRect = boundingRect ( squares[iLargestContourIndex] );

	// Sort detected boxes
	for (auto &vp: contours_poly) {
		blowMe (vp);
	}

	// Apply crop to the original image, and don't return to caller the changes
	// we've made to the image, such as de-noize etc.
	Mat src;
	timg.copyTo(src);

	if ( contours_poly[0].size() == 4 ) {

		std::vector<Point2f> quad_pts;
		std::vector<Point2f> squre_pts;
		quad_pts.push_back ( Point2f ( contours_poly[0][ TOP_LEFT     ].x, contours_poly[0][ TOP_LEFT     ].y) );
		quad_pts.push_back ( Point2f ( contours_poly[0][ BOTTOM_LEFT  ].x, contours_poly[0][ BOTTOM_LEFT  ].y) );
		quad_pts.push_back ( Point2f ( contours_poly[0][ TOP_RIGHT    ].x, contours_poly[0][ TOP_RIGHT    ].y) );
		quad_pts.push_back ( Point2f ( contours_poly[0][ BOTTOM_RIGHT ].x, contours_poly[0][ BOTTOM_RIGHT ].y) );

		squre_pts.push_back ( Point2f ( boundRect.x,boundRect.y) );
		squre_pts.push_back ( Point2f ( boundRect.x,boundRect.y+boundRect.height) );
		squre_pts.push_back ( Point2f ( boundRect.x+boundRect.width,boundRect.y) );
		squre_pts.push_back ( Point2f ( boundRect.x+boundRect.width,boundRect.y+boundRect.height) );

		Mat transmtx    = getPerspectiveTransform(quad_pts,squre_pts);
		Mat transformed = Mat::zeros(src.rows, src.cols, CV_8UC3);
		warpPerspective ( src, transformed, transmtx, src.size() );

		// calculate the logo area in order to ignore during mark extraction
		int iCouponWidth = abs ( contours_poly[0][0].x - contours_poly[0][1].x );
		float logo_width = iCouponWidth*0.11; //Logo area is 11% of the given coupon width
		m_xDistance      = logo_width * (iCouponWidth/(boundRect.width*1.0) );
		qDebug () << "m_xDistance: " << m_xDistance;


		// crop the original image around the detected rectangular and also crop the logo area
		// along with the QR barcode

        if(boundRect.x >= 0 && boundRect.y >= 0 && boundRect.width + boundRect.x < transformed.cols && boundRect.height + boundRect.y < transformed.rows)
        {
            boundRect.x     = boundRect.x + m_xDistance - 15;
            boundRect.width = boundRect.width - m_xDistance + 15;
            croppedImage	= transformed(boundRect);

        }
        else{
            int border=100;
            copyMakeBorder(transformed,transformed,0,border,0,border,BORDER_REPLICATE);
            boundRect.x     = boundRect.x + m_xDistance - 15;
            boundRect.width = boundRect.width - m_xDistance + 15;
            croppedImage	= transformed(boundRect);
        }

//        std::cout << boundRect.y << ", " << boundRect.size() << ", " << transformed.cols << ", " << transformed.rows << std::endl;
//        boundRect.x     = boundRect.x+m_xDistance-15;
//        boundRect.width = boundRect.width-m_xDistance+15;

//        std::cout << boundRect.y << ", " << boundRect.size() << ", " << transformed.cols << ", " << transformed.rows << std::endl;
//        croppedImage	= transformed(boundRect);


#if defined DEBUG_BETBOXES_DEEP
		Point P1=contours_poly[0][0];
		Point P2=contours_poly[0][1];
		Point P3=contours_poly[0][2];
		Point P4=contours_poly[0][3];


		line(src,P1,P2, Scalar(0,0,255),1,CV_AA,0);
		line(src,P2,P3, Scalar(0,0,255),1,CV_AA,0);
		line(src,P3,P4, Scalar(0,0,255),1,CV_AA,0);
		line(src,P4,P1, Scalar(0,0,255),1,CV_AA,0);
		rectangle ( src, boundRect,Scalar(0,255,0),1,8,0 );
		rectangle ( transformed, boundRect,Scalar(0,255,0),1,8,0 );


		imshow("quadrilateral", transformed);
		imshow("src",           src);
        imshow("cropped",       croppedImage);
        waitKey();
#endif

	} else {

		qDebug () << "Make sure that your are getting 4 corner using approxPolyDP. Corners found: " <<  contours_poly[0].size();
	}

	return croppedImage;

}



/**************************************************************
* Input: Mat image
* Output: vector<Point> interpolated,
*         vector<vector <Point> > RowsCols
*
* Description: Process input image and return the coordinates of
* the detected and sorted bet box centers
* ************************************************************/
vector<Point> patternDetection::detectBoxes ( Mat image )
{
	vector<Point> interpolated;
	vector<vector<Point> > squares;
	vector<vector<Point> > contours;
	vector<Point> approx;
	size_t i = 0;

	Point rightFiducial {0,0};
	Mat pyr, timg;


	// down-scale and upscale the image to filter out the noise. When we do so the number
	// of contours reduces substantially resulting in faster code because we got a smaller
	// contours vector to deal with.
	if ( m_bSpeedUpDetection ) {

		pyrDown ( Mat(image), pyr);
		pyrUp   ( pyr, timg );

	} else {

		timg = image.clone ();
	}


	cvtColor ( timg, timg, CV_BGR2GRAY );
	cv::Mat tmp;
	GaussianBlur ( timg, tmp, cv::Size(5,5), 5 );
	addWeighted  ( timg, 1.5, tmp, -0.5, 0, timg );

	adaptiveThreshold ( timg, timg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2 /*Works better than 9, 5*/);

	findContours ( timg, contours, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS );

	for ( i = 0; i < contours.size(); i++ ) {

		approx.clear();
		approxPolyDP ( Mat(contours[i]), approx, m_iEpsilon/*arcLength(Mat(contours[i]), true)*0.07*/, true );

		if ( approx.size() == 4
			 && fabs( contourArea ( Mat(approx) ) ) > BOX_MIN_SIZE
			 && fabs( contourArea ( Mat(approx) ) ) < BOX_MAX_SIZE
			 && isContourConvex   ( Mat(approx) ) )
		{
			double maxCosine = 0;

			for( int j = 2; j < 5; j++ ){

				// find the maximum cosine of the angle between joint edges
				double cosine = fabs(calculateAngle(approx[j%4], approx[j-2], approx[j-1]));
				maxCosine = MAX(maxCosine, cosine);
			}

			if ( maxCosine < 0.3 ) {
				squares.push_back(approx);
			}
		}
	}


	if ( squares.size() == 0 ) {

		return interpolated;
	}

	for ( auto &vp : squares ) {

		blowMe(vp);
	}

	if ( squares.size() == 0 ) {

		qDebug () << "The corners were not sorted correctly! Missing coupon information";
		return interpolated;
	}


	//Remove duplicate detected squares
	int offset   = 8;
	int removals = 0;

	for ( i = 0; i < squares.size(); ++i ) {

		int currentx = squares[i][0].x;
		int currenty = squares[i][0].y;

		for ( size_t j = i+1; j < squares.size(); ++j ) {

			int tempx = squares[j][0].x;
			int tempy = squares[j][0].y;

			if ( abs(currentx - tempx)<=offset && abs(currenty - tempy)<=offset ) {

				if ( currentx >= tempx ) {

					vector<vector<Point> >::iterator iter = squares.begin() + j;
					squares.erase(iter);
					j--;

				} else {

					vector<vector<Point> >::iterator iter = squares.begin() + i;
					squares.erase(iter);
					i--;
				}
				++removals;
			}
		}
	}



    if ( m_bPreviewProcessed ) {

        Mat boxedImage;
		image.copyTo(boxedImage);
		for ( size_t i = 0; i < squares.size(); ++i ) {

			const Point* p = &squares[i][0];
			int n = (int)squares[i].size();
			polylines(boxedImage, &p, &n, 1, true, Scalar(0,255,0), 1, LINE_AA);
		}

		imshow("image with boxes", boxedImage);
        waitKey();
    }

#if defined DEBUG_BETBOXES
    Mat boxedImage;
	image.copyTo(boxedImage);
	for ( size_t i = 0; i < squares.size(); ++i ) {

		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		polylines(boxedImage, &p, &n, 1, true, Scalar(0,255,0), 1, LINE_AA);
	}
	imshow("image with boxes", boxedImage);
    waitKey();
#endif



	//Search for any missing boxes and add them
	interpolated = interpolateBoxes ( squares );

	//we need to short boxes based on x and y coordinate and coupon orientation
	offset = 3;
	Point local2;
	for ( i = 0; i < interpolated.size()-1; i++ ) {

		for ( size_t j = i+1; j< interpolated.size(); j++ ) {

			if ( (interpolated[i].x >= interpolated[j].x-offset)  ) {

				local2=interpolated[i];
				interpolated[i] = interpolated[j];
				interpolated[j] = local2;
			}
		}
	}

	//Remove fiducials and false detections from detected boxes (100=distance of fiducial and bet areas)
	for ( size_t x = 0; x < interpolated.size()-1; x++ ) {


		if ( interpolated[x].x < rightFiducial.x /*+ m_xDistance*/ ) {

			vector<Point> ::iterator iter = interpolated.begin() + x;
			interpolated.erase(iter);
			x--;
		}
	}

	return interpolated;
}

/**************************************************************
* Input: vector<Point> expectedCenters, vector<Point> foundBoxes
* Output: vector<Point> boxCenters,
*
* Description: Apply calculations in order to interpolate possible
* undetected boxes based on coupon definition file
* ************************************************************/
vector<Point> patternDetection::interpolateBoxes ( vector<vector<Point>> foundBoxes )
{

	//We need to get box centers at first
	vector<Point> boxCenters;

	for ( auto &vp : foundBoxes ) {

		boxCenters.push_back ( detectCenters(vp) );
	}

	return boxCenters;

}

/**************************************************************
* Input: Mat areaROI, vector<Point> boxCenters,
* vector<vector> > RowsCols
* Output: vector<Point> markedBoxes
*
* Description: Examine ROI of each detected box and calculate if
* a betmark exists
* ************************************************************/
QList<int> patternDetection::ROIaveragePixelValue ( Mat image, vector<Point> boxCenters )
{
	vector<Point> markedBoxes;
	QList<int> indexedBoxes;
	size_t i = 0;

	Mat imageOut  = image.clone();
	cvtColor(image,image,CV_BGR2GRAY);
	GaussianBlur(image, image, cv::Size(5, 5), 5);
	adaptiveThreshold(image, image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 75, 25 ); //NONDAS: with C basically you can control the sensitivity of marks to be found. Better control it via GUI.
	bitwise_not(image, image);

	if ( !image.data ) {

		qDebug () << "Failed to load ROI of image";
		return indexedBoxes;
	}

	for ( i = 0; i < boxCenters.size(); ++i ) {

		Mat betROI = image ( Rect ( boxCenters[i].x-3, boxCenters[i].y-5, 6, 10 ) ); //filled point
		double avgPixel =(double)countNonZero(betROI)/(betROI.size().width*betROI.size().height);

		if ( avgPixel >= 0.25 ) {

#ifdef DEBUG
			qDebug () << "AVG = " << avgPixel << "@(" << i << ")";
#endif
			markedBoxes.push_back(boxCenters[i]);
			indexedBoxes.push_back(i);
		}
	}

    m_bPreviewProcessed=true;
	if ( m_bPreviewProcessed ) {


		for ( i = 0; i < markedBoxes.size(); ++i ) {

			circle( imageOut, markedBoxes[i], 8, Scalar(0,0,255), 2 );
		}

		imshow ( "markedboxes", imageOut );
        waitKey();
	}

	return indexedBoxes;
}

/**************************************************************
* Input: vector<Point>& corners, Point center
* Output:
*
* Description: Sort the four detected cornes of each bet box in
* reference of the box center
* ************************************************************/
void patternDetection::sortCorners ( vector<Point>& corners, Point center )
{
	std::vector<cv::Point> top, bot;
	size_t i = 0;

	for ( i = 0; i < corners.size(); ++i ) {

		if ( corners[i].y < center.y ) {

			top.push_back( corners[i] );

		} else {

			bot.push_back( corners[i] );
		}
	}

#ifdef DEBUG
	if ( top.size()!=2 || bot.size()!=2 ) {

		qDebug () << top.size() << " - " << bot.size();

		qDebug () << "Center" << ": " << center.x << "-" << center.y;

		for ( i = 0; i < bot.size(); ++i ) {

			qDebug () << "Bottom[" << i << "]" << ": " << bot[i].x << "-" << bot[i].y;
		}

		for ( i = 0; i < top.size(); ++i ) {

			qDebug () << "Top[" << i << "]" << ": " << top[i].x << "-" << top[i].y;
		}
	}
#endif

	corners.clear();

	if (top.size() == 2 && bot.size() == 2){

		cv::Point tl = top[0].x > top[1].x ? top[1] : top[0];
		cv::Point tr = top[0].x > top[1].x ? top[0] : top[1];
		cv::Point bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
		cv::Point br = bot[0].x > bot[1].x ? bot[0] : bot[1];


		corners.push_back(tl);
		corners.push_back(tr);
		corners.push_back(br);
		corners.push_back(bl);
	}
}

/**************************************************************
* Input: vector<Point>& corners
* Output:
*
* Description: Sort the detected boxes
* ************************************************************/
void patternDetection::blowMe ( vector<Point>& corners )
{

	auto center ( this->m_Center );

	// Get mass center
	for ( size_t i = 0; i < corners.size(); ++i ) {

		center += corners[i];
	}

	center *= ( 1. / corners.size() );

	sortCorners ( corners, center );
}

/**************************************************************
* Input: vector<Point>& corners
* Output: Point center
*
* Description: Calculate the detected object mass center
* ************************************************************/
Point patternDetection::detectCenters ( vector<Point>& corners )
{
	auto center ( this->m_Center );

	// Get mass center
	for ( size_t i = 0; i < corners.size(); ++i ) {

		center += corners[i];
	}

	center *= (1. / corners.size());

	return center;
}


