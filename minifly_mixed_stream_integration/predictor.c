#include <math.h>
#include <stdint.h>
#include <string.h>

void advance(const uint8_t *bytes, const int32_t *features, int n, int width,
             float *weights, float *bias, int hashed, int train, double *losses) {
    const float rates[7] = {.03f,.03f,.03f,.10f,.10f,.10f,.10f};
    for (int i=0;i<n;i++) {
        const int32_t *ids=features+i*width;
        double logits[256], probabilities[256], total=0, max=-INFINITY;
        int count=0,y=bytes[i];
        while (count<width && ids[count]>=0) count++;
        for (int j=0;j<256;j++) {
            double z=hashed?bias[j]:0;
            for (int k=0;k<count;k++) z+=weights[256*ids[k]+j];
            logits[j]=hashed?z:(float)z;
            if (logits[j]>max) max=logits[j];
        }
        for (int j=0;j<256;j++) {
            double z=hashed?exp(logits[j]-max):fmax(logits[j],.001f);
            probabilities[j]=z;
            total+=z;
        }
        losses[i]=-log2(fmax(probabilities[y]/total,1e-12));
        if (!train) continue;
        for (int j=0;j<256;j++) {
            float e=(float)(hashed?probabilities[j]/total:logits[j])-(j==y);
            if (hashed) bias[j]-=.001f*e;
            for (int k=0;k<count;k++) {
                float delta=hashed?rates[k]*e:(float)(.1*(double)e/count);
                weights[256*ids[k]+j]-=delta;
            }
        }
    }
}
