#include <gtest/gtest.h>
#include <fstream>
#include <iterator>
#include "data/tensor_bundle.h"
TEST(TensorBundleCrossLanguageTest, DecodesPythonGoldenWhenProvided){const char*p=std::getenv("PBE_PYTHON_BUNDLE_GOLDEN");if(!p)GTEST_SKIP()<<"set PBE_PYTHON_BUNDLE_GOLDEN";std::ifstream f(p,std::ios::binary);std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});data::TensorBundleSchema s;std::string reason;ASSERT_EQ(data::DecodeTensorBundlePayload(b.data(),b.size(),&s,&reason),data::DataError::kOk)<<reason;ASSERT_EQ(s.components.size(),2u);EXPECT_EQ(s.components[0].name,"image_features");EXPECT_EQ(s.components[1].name,"image_grid_thw");}
