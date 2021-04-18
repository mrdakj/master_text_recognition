#include <fdeep/fdeep.hpp>
#include "include/image.h"

int main()
{
    const auto model = fdeep::load_model("../model/fdeep_model.json", true, fdeep::dev_null_logger);

    int good = 0;
    int all = 0;

    for(auto& p: fs::directory_iterator("a")) {
        ++all;
        auto image_name = p.path().string();
        char letter = image_name[0];
        // std::cout << letter << std::endl;
        image img(image_name);
        img.resize(28,28);

        auto t = img.get_tensor(0,1);

        // std::cout << fdeep::show_tensor_shape(t.shape());
        // std::cout << fdeep::show_tensor(t) << std::endl;
        // const auto result = model.predict({t});
        // std::cout << fdeep::show_tensors(result) << std::endl;

        const auto result_class = model.predict_class({t});
        char result = 97+result_class;
        if (result == letter) {
            ++good;
        }
        // std::cout << result << std::endl;
    }

    std::cout << (double)good/(double)all << std::endl;
}
