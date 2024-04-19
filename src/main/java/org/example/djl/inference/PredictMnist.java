package org.example.djl.inference;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PredictMnist {
    private static final Logger logger = LoggerFactory.getLogger(PredictMnist.class);

    public static void main(String[] args) {
        Classifications classifications = null;
        try {
            classifications = PredictMnist.predict();
            System.out.println(classifications);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (ModelException e) {
            throw new RuntimeException(e);
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
        logger.info("{}", classifications);
    }

    public static Classifications predict() throws IOException, ModelException, TranslateException {

        //get the image/file to run inference on
        Path imageFile = Paths.get("src/main/resources/images/0.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String modelName = "mlp";
        try (Model model = Model.newInstance(modelName)) {
            model.setBlock(new Mlp(28 * 28, 10, new int[] {128, 64}));

            // load a local model from build/model folder within the project.
            Path modelDir = Paths.get("build/model");
            model.load(modelDir);

            List<String> classes =
                    IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
            Translator<Image, Classifications> translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new ToTensor())
                            .optSynset(classes)
                            .optApplySoftmax(true)
                            .build();

            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                return predictor.predict(img);
            }
        }
    }
}
