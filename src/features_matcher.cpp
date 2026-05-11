#include "features_matcher.h"

#include <iostream>
#include <map>

FeatureMatcher::FeatureMatcher(cv::Mat intrinsics_matrix, cv::Mat dist_coeffs,
                               bool use_modern_features, double focal_scale) :
  use_modern_features_(use_modern_features)
{
  intrinsics_matrix_ = intrinsics_matrix.clone();
  dist_coeffs_ = dist_coeffs.clone();
  new_intrinsics_matrix_ = intrinsics_matrix.clone();
  new_intrinsics_matrix_.at<double>(0,0) *= focal_scale;
  new_intrinsics_matrix_.at<double>(1,1) *= focal_scale;
}

cv::Mat FeatureMatcher::readUndistortedImage(const std::string& filename )
{
  cv::Mat img = cv::imread(filename), und_img, dbg_img;
  cv::undistort	(	img, und_img, intrinsics_matrix_, dist_coeffs_, new_intrinsics_matrix_ );

  return und_img;
}

void FeatureMatcher::extractFeatures()
{
  features_.resize(images_names_.size());
  descriptors_.resize(images_names_.size());
  feats_colors_.resize(images_names_.size());

  auto orb_detector = cv::ORB::create(10000, 1.2, 8);

  for( int i = 0; i < images_names_.size(); i++  )
  {
    std::cout<<"Computing descriptors for image "<<i<<std::endl;
    cv::Mat img = readUndistortedImage(images_names_[i]);


    //////////////////////////// Code to be completed (2/7) /////////////////////////////////
    // Extract salient points + descriptors from i-th image.
    //
    // A standard implementation (else branch) that uses the ORB features is already provided.
    // It stores them into the features_[i] and descriptors_[i] vectors, and extract the
    // color (cv::Vec3b) of each feature and store in feats_colors_[i] vector.
    //
    // You are required to implement an alternative, more modern feature detection and
    // description scheme inside the if (use_modern_features_) branch (e.g., by means the
    // loadExternalFeatures() function). Examples are SuperPoint
    // (https://github.com/eric-yyjau/pytorch-superpoint), DISK
    // (https://github.com/cvlab-epfl/disk) or ALIKED (https://github.com/Shiaoming/ALIKED),
    // or other alternatives.
    //
    // IMPORTANT: You also need to update the matching part, see the branch
    // if (use_modern_features_) in the FeatureMatcher::exhaustiveMatching() method.
    // For some methods, the feature description and matching phase are merged,
    // so you may only need to change Feature Matcher::exhaustive Matching()

    if (use_modern_features_)
    { 
      std::string img_path = images_names_[i];
      
      size_t last_slash_idx = img_path.find_last_of("\\/");
      std::string filename = img_path.substr(last_slash_idx + 1);
      std::string method_folder = "aliked_2"; // e.g., "aliked", "xfeat", "dedode"
      std::string feature_file = "../datasets/" + method_folder + "/" + filename + ".yaml"; 

      cv::FileStorage fs(feature_file, cv::FileStorage::READ);
      
    if(fs.isOpened()) {
          cv::Mat kpts_mat;
          fs["keypoints"] >> kpts_mat;
          fs["descriptors"] >> descriptors_[i];
          fs.release();
          std::vector<cv::Point2f> distorted_pts;
          for (int r = 0; r < kpts_mat.rows; ++r) {
              distorted_pts.push_back(cv::Point2f(kpts_mat.at<float>(r, 0), kpts_mat.at<float>(r, 1)));
          }
          std::vector<cv::Point2f> undistorted_pts;
          cv::undistortPoints(distorted_pts, undistorted_pts, intrinsics_matrix_, dist_coeffs_, cv::noArray(), new_intrinsics_matrix_);
          features_[i].clear();
          for (const auto& pt : undistorted_pts) {
              features_[i].push_back(cv::KeyPoint(pt.x, pt.y, 1.0f));
          }
          // ----------------------------

      } else {
          std::cerr << "\nCould not load features from: " << feature_file << std::endl;
          exit(EXIT_FAILURE); 
      }

      // Look-up features colors
      feats_colors_[i].reserve(features_[i].size());
      for( auto &f : features_[i])
      {
        int x = std::min(std::max(cvRound(f.pt.x), 0), img.cols - 1);
        int y = std::min(std::max(cvRound(f.pt.y), 0), img.rows - 1);
        feats_colors_[i].emplace_back(img.at<cv::Vec3b>(y, x));
      }
    }
    else
    {
      // Standard ready to use ORB implementation
      orb_detector->detectAndCompute(img, cv::Mat(), features_[i], descriptors_[i]);

      // Look-up features colors
      feats_colors_[i].reserve(features_[i].size());
      for( auto &f : features_[i])
      {
        feats_colors_[i].emplace_back(img.at<cv::Vec3b>(f.pt.y, f.pt.x));
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////
  }
}

void FeatureMatcher::exhaustiveMatching()
{
  std::vector<cv::DMatch> matches, inlier_matches;
  
  for( int i = 0; i < images_names_.size() - 1; i++ )
  {
    for( int j = i + 1; j < images_names_.size(); j++ )
    {
      std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;
      std::vector<cv::DMatch> matches, inlier_matches;

      if( use_modern_features_ )
      {
        // Modern descriptors (SuperPoint, etc.) are usually float matrices.
        // You may use a BruteForce with L2 distance and enable cross-check for better precision.
        //
        // You could also use a modern Matching Network such as SuperGlue/LightGlue
        // (https://github.com/magicleap/supergluepretrainednetwork) subsequent
        // geometric verification (Code to be completed (1/7)) is not required,
        // since these networks perform both matching and geometric verification.
        // In this case, you may follow OPTION A or OPTION A (see above).
        /////////////////////////////////////////////////////////////////////////////////////////

        auto matcher = cv::BFMatcher::create(cv::NORM_L2, true);
        //for models like XFeat
        descriptors_[i].convertTo(descriptors_[i], CV_32F);
        descriptors_[j].convertTo(descriptors_[j], CV_32F); 
        matcher->match(descriptors_[i], descriptors_[j], matches);
        /////////////////////////////////////////////////////////////////////////////////////////

      }
      else
      {
        std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;
        auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        matcher->match(descriptors_[i], descriptors_[j], matches);
      }

      //////////////////////////// Code to be completed (1/7) /////////////////////////////////
      // Perform Geometric Verification of matches, possibly discarding the outliers
      // (remember that features have been extracted from undistorted images that now has
      // new_intrinsics_matrix_ as K matrix and no distortions).
      // As geometric models, use both the Essential matrix and the Homograph matrix,
      // both by setting new_intrinsics_matrix_ as K matrix.
      // As threshold in the functions to estimate both models, you may use 1.0 or similar.
      // Store inlier matches into the inlier_matches vector
      // Do not set matches between two images if the amount of inliers matches
      // (i.e., geomatrically verified matches) is small (say <= 5 matches)
      // In case of success, set the matches with the function:
      //
      // setMatches( i, j, inlier_matches);
      //
      // where i,j matched images indices.
      /////////////////////////////////////////////////////////////////////////////////////////
      
   //////////////////////////// Code to be completed (1/7) /////////////////////////////////
      if (matches.size() > 5) 
      {
          std::vector<cv::Point2f> pts1, pts2;
          
          for (const auto& m : matches) {
              pts1.push_back(features_[i][m.queryIdx].pt);
              pts2.push_back(features_[j][m.trainIdx].pt);
          }

          cv::Mat mask_E, mask_H;
          double inlier_threshold = 1.0;
          
          cv::findEssentialMat(pts1, pts2, new_intrinsics_matrix_, cv::USAC_MAGSAC, 0.99, inlier_threshold, mask_E);
          cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, inlier_threshold, mask_H);

          for (size_t k = 0; k < matches.size(); k++) {
              bool is_inlier_E = !mask_E.empty() && mask_E.at<uchar>(k) == 1;
              bool is_inlier_H = !mask_H.empty() && mask_H.at<uchar>(k) == 1;

              if (is_inlier_E || is_inlier_H) {
                  inlier_matches.push_back(matches[k]);
              }
          }

          if (inlier_matches.size() > 5) {
              setMatches(i, j, inlier_matches);
          }
      }
      /////////////////////////////////////////////////////////////////////////////////////////

      /////////////////////////////////////////////////////////////////////////////////////////
    }
  }
}

void FeatureMatcher::writeToFile ( const std::string& filename, bool normalize_points ) const
{
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    std::cerr << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_poses_, num_points_, num_observations_);

  double *tmp_observations;
  cv::Mat dst_pts;
    if(normalize_points)
    {
      cv::Mat src_obs( num_observations_,1, cv::traits::Type<cv::Vec2d>::value,
                      const_cast<double *>(observations_.data()));
      cv::undistortPoints(src_obs, dst_pts, new_intrinsics_matrix_, cv::Mat());
      tmp_observations = reinterpret_cast<double *>(dst_pts.data);
    }
    else
    {
      tmp_observations = const_cast<double *>(observations_.data());
    }

  for (int i = 0; i < num_observations_; ++i)
  {
    fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", tmp_observations[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  if( colors_.size() == 3*num_points_ )
  {
    for (int i = 0; i < num_points_; ++i)
      fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
  }

  fclose(fptr);
}

void FeatureMatcher::testMatches( double scale )
{
  // For each pose, prepare a map that reports the pairs [point index, observation index]
  std::vector< std::map<int,int> > cam_observation( num_poses_ );
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    int i_cam = pose_index_[i_obs], i_pt = point_index_[i_obs];
    cam_observation[i_cam][i_pt] = i_obs;
  }

  for( int r = 0; r < num_poses_; r++ )
  {
    for (int c = r + 1; c < num_poses_; c++)
    {
      int num_mathces = 0;
      std::vector<cv::DMatch> matches;
      std::vector<cv::KeyPoint> features0, features1;
      for (auto const &co_iter: cam_observation[r])
      {
        if (cam_observation[c].find(co_iter.first) != cam_observation[c].end())
        {
          features0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1], 0.0);
          features1.emplace_back(observations_[2*cam_observation[c][co_iter.first]],observations_[2*cam_observation[c][co_iter.first] + 1], 0.0);
          matches.emplace_back(num_mathces,num_mathces, 0);
          num_mathces++;
        }
      }
      cv::Mat img0 = readUndistortedImage(images_names_[r]),
          img1 = readUndistortedImage(images_names_[c]),
          dbg_img;

      cv::drawMatches(img0, features0, img1, features1, matches, dbg_img);
      cv::resize(dbg_img, dbg_img, cv::Size(), scale, scale);
      cv::imshow("", dbg_img);
      if (cv::waitKey() == 27)
        return;
    }
  }
}

void FeatureMatcher::setMatches( int pos0_id, int pos1_id, const std::vector<cv::DMatch> &matches )
{

  const auto &features0 = features_[pos0_id];
  const auto &features1 = features_[pos1_id];

  auto pos_iter0 = pose_id_map_.find(pos0_id),
      pos_iter1 = pose_id_map_.find(pos1_id);

  // Already included position?
  if( pos_iter0 == pose_id_map_.end() )
  {
    pose_id_map_[pos0_id] = num_poses_;
    pos0_id = num_poses_++;
  }
  else
    pos0_id = pose_id_map_[pos0_id];

  // Already included position?
  if( pos_iter1 == pose_id_map_.end() )
  {
    pose_id_map_[pos1_id] = num_poses_;
    pos1_id = num_poses_++;
  }
  else
    pos1_id = pose_id_map_[pos1_id];

  for( auto &match:matches)
  {

    // Already included observations?
    uint64_t obs_id0 = poseFeatPairID(pos0_id, match.queryIdx ),
        obs_id1 = poseFeatPairID(pos1_id, match.trainIdx );
    auto pt_iter0 = point_id_map_.find(obs_id0),
        pt_iter1 = point_id_map_.find(obs_id1);
    // New point
    if( pt_iter0 == point_id_map_.end() && pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = num_points_++;
      point_id_map_[obs_id0] = point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);

      // Average color between two corresponding features (suboptimal since we shouls also consider
      // the other observations of the same point in the other images)
      cv::Vec3f color = (cv::Vec3f(feats_colors_[pos0_id][match.queryIdx]) +
                        cv::Vec3f(feats_colors_[pos1_id][match.trainIdx]))/2;

      colors_.push_back(cvRound(color[2]));
      colors_.push_back(cvRound(color[1]));
      colors_.push_back(cvRound(color[0]));

      num_observations_++;
      num_observations_++;
    }
      // New observation
    else if( pt_iter0 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id1];
      point_id_map_[obs_id0] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      num_observations_++;
    }
    else if( pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id0];
      point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);
      num_observations_++;
    }
//    else if( pt_iter0->second != pt_iter1->second )
//    {
//      std::cerr<<"Shared observations does not share 3D point!"<<std::endl;
//    }
  }
}
void FeatureMatcher::reset()
{
  point_index_.clear();
  pose_index_.clear();
  observations_.clear();
  colors_.clear();

  num_poses_ = num_points_ = num_observations_ = 0;
}
