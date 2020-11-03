Test flow field
------------------------------------------

simulation domain size 256 x 256
0 - 999 frames
dt = 1.0 s

Example of yaml files processing in OpenCV
------------------------------------------

for ( int i = 0; i < 1000; ++i )
{
	char file_name[128];
	sprintf( file_name, "u%05d.yml", i );
	cv::Mat u;
	cv::FileStorage fs( file_name, cv::FileStorage::READ );
	fs["flow"] >> u;

	// TODO visualization of vector field u  
  //const cv::Point2f velocity_at_xy = u.at<cv::Point2f>( cv::Point( x, y ) );
}
