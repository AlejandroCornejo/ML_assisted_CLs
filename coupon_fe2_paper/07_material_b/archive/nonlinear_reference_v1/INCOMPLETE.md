# Incomplete orchestration attempt

This invocation exited with TypeError before recording any extended FOM state:
dict() got multiple values for keyword argument 'path' in baseline metadata.
The historical reference rows already contain path, unlike the check rows.
This is a driver record-construction bug, not a mechanical nonconvergence.
The original report and driver snapshot are retained; exclude this invocation
from numerical comparisons. The corrected reference repeat is nonlinear_reference_v2.
