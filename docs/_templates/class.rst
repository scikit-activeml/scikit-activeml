{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item != '__init__' and is_skactiveml_method(fullname, item) %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% for item in methods %}
   {%- if item != '__init__' and not is_skactiveml_method(fullname, item) %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. _sphx_glr_backref_{{fullname}}:

.. minigallery:: {{fullname}}
   :add-heading:
