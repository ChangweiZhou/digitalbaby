"""Post-pilot API-only revision; frozen main kernel remains in core.py.

No new learning rule or answer route. Numeric pilot results belong to core.py.
"""
from core import Fly as PilotFly
class Fly(PilotFly):
 @staticmethod
 def _payload(value,limit):
  value=bytes(value)
  if len(value)>limit:raise ValueError('payload cap')
  return value
 def query(self,prompt,tape):
  return super().query(self._payload(prompt,256),tape)
 def generation(self,tape,cap=16):
  if not isinstance(cap,int) or not 0<=cap<=16:raise ValueError('generation cap')
  return super().generation(tape,cap)
 def record(self,prompt,answer,native=True,controller=True,tape=0):
  prompt=self._payload(prompt,256);answer=self._payload(answer,16)
  return super().record(prompt,answer,native,controller,tape)
