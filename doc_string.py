#%%
import model_prep as mp
import inspect
#%%
def print_doc_str(class_list):
    for c in class_list:
        print()
        print(c.__name__)
        print(inspect.getdoc(c))
        method_list = [m for m in dir(c) if not m.startswith('_')]
        for m in method_list:
            attr_ = getattr(c,m)
            print()
            print(attr_.__name__)
            print(inspect.getdoc(attr_))
            print()

class_list = [mp.ModelPrepTransFeatures,mp.ModelPrepSelectFeatures]
print_doc_str(class_list)
#%%
import putilities as pu
class_list = [pu.ConvertDtype]
print_doc_str(class_list)

