#include <fdeep/fdeep.hpp>
#include "include/image.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "usage: ./main letter" << std::endl;
        return -1;
    }

    std::string letter = argv[1];

    // const auto model_one_two = fdeep::load_model("../model/one_two.json", true, fdeep::dev_null_logger);
    const auto model_one = fdeep::load_model("../model/one.json", true, fdeep::dev_null_logger);
    // const auto model_first = fdeep::load_model("../model/first.json", true, fdeep::dev_null_logger);
    // const auto model_second = fdeep::load_model("../model/second.json", true, fdeep::dev_null_logger);

    int all = 0;

    // std::vector<int> output_class_one_two(2,0);
    std::vector<int> output_class_one(26,0);

    for(auto& p: fs::directory_iterator(letter)) {
        ++all;
        auto image_name = p.path().string();
        image img(image_name);
        img.resize(28,28);

        auto t = img.get_tensor(0,1);

        const auto result_class_one = model_one.predict_class({t});
        ++output_class_one[result_class_one];

        // const auto probs_one = model_one.predict({t}).front();
        // std::vector<float> vec = probs_one.to_vector();
        // float prob_one = probs_one.get(fdeep::tensor_pos(result_class_one));
        // std::cout << fdeep::show_tensor_shape(probs_one.shape()) << std::endl;
        // std::cout << fdeep::show_tensor(probs_one) << std::endl;
    }

    // std::cout << "all " << all << std::endl;
    // std::cout << "one " << output_class_one_two[0] << " " << output_class_one_two[0]/(double)all << std::endl;
    // std::cout << "two " << output_class_one_two[1] << " " << output_class_one_two[1]/(double)all << std::endl;

    for (int i = 0; i < 26; i++) {
        std::cout << (char)(97+i) << " ";
        std::cout << 100*((double)output_class_one[i]/(double)all) << " ";
        std::cout << std::endl;
    }
}
